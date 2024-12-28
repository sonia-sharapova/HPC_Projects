#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <stdint.h>
#include <time.h>
#include <cblas.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

// CUDA and CUBLAS functions
//#include <helper_functions.h>
//#include <helper_cuda.h>

#define MAX(a,b) ((a)<(b)?(b):(a))

#define TRAIN_IMG "../Data/train-images-idx3-ubyte"
#define TRAIN_LBL "../Data/train-labels-idx1-ubyte"
#define TEST_IMG "../Data/t10k-images-idx3-ubyte"
#define TEST_LBL "../Data/t10k-labels-idx1-ubyte"

float totalTime = 0.0;
float avgTime = 0.0;

float alpha;
int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

/* MNIST read-in code obtained from Tarun Vallabhaneni */
float* readMNISTLabels(const char* filename, int* number_of_labels) {
    FILE* file = fopen(filename, "rb");
    if (file) {
        int magic_number = 0;
        fread(&magic_number, sizeof(magic_number), 1, file);
        magic_number = reverseInt(magic_number);

        int num_labels = 0;
        fread(&num_labels, sizeof(num_labels), 1, file);
        num_labels = reverseInt(num_labels);

        *number_of_labels = num_labels;

        // Allocating a 1D array instead of a double pointer
        // Size is num_labels * 10 because each label is a one-hot encoded vector of size 10
        float* labels = (float*)calloc(num_labels * 10, sizeof(float));

        for (int i = 0; i < num_labels; i++) {
            unsigned char temp = 0;
            fread(&temp, sizeof(temp), 1, file);
            // Calculating the index in the 1D array where the one-hot encoding starts for this label
            // Then setting the appropriate position to 1.0
            labels[i * 10 + temp] = 1.0;
        }
        fclose(file);
        return labels;
    } else {
        printf("Error opening file\n");
        return NULL;
    }
}


float* readMNISTImages(const char* filename, int* number_of_images, int* size) {
    FILE* file = fopen(filename, "rb");
    if (file) {
        int magic_number = 0;
        int num_images = 0;
        int rows = 0;
        int cols = 0;
        fread(&magic_number, sizeof(magic_number), 1, file);
        magic_number = reverseInt(magic_number);
        fread(&num_images, sizeof(num_images), 1, file);
        num_images = reverseInt(num_images);
        fread(&rows, sizeof(rows), 1, file);
        rows = reverseInt(rows);
        fread(&cols, sizeof(cols), 1, file);
        cols = reverseInt(cols);

        *number_of_images = num_images;
        *size = rows * cols;

        // Allocate memory for all images
        float* images = (float*)malloc(num_images * rows * cols * sizeof(float));

        // Read each image and normalize
        for(int i = 0; i < num_images; i++) {
            for(int j = 0; j < rows * cols; j++) {
                unsigned char temp = 0;
                fread(&temp, sizeof(temp), 1, file);
                // Normalize pixel values to the range [0, 1]
                images[i * rows * cols + j] = temp / 255.0;
            }
        }
        fclose(file);
        return images;
    } else {
        printf("Error opening file\n");
        return NULL;
    }
}

struct layer {
	int batch, in, out;
	float *a_list;  // activations (batch*in)
	float *z_list;  // A*z_list + bias (batch*out)
	float *fwd;     // to connect no next layer's a_list (batch*out)

	float *deltas;  // (batch*out)
	float *bwd;     // to connect to prev layer's deltas (batch*int)

	float *biases;  // bias vector (out)
	float *weights; // weight matrix (in * out)
	float *w_grad;  // weight gradient matrix (in * out)
};

struct network {
	int L;
	int batch_size;
	struct layer *layer; /* [L] */
};

void setup_layer(struct layer *layer, int Batch, int In, int Out) {

	layer->batch = Batch;
	layer->in = In;
	layer->out = Out;
	
	layer->a_list = calloc((Batch*In), sizeof *layer->a_list);
	layer->z_list = calloc((Batch*Out), sizeof *layer->z_list);
	layer->biases = calloc((Batch*Out), sizeof *layer->biases);
	layer->weights = calloc((In*Out), sizeof *layer->weights);
	layer->w_grad = calloc((In*Out), sizeof *layer->w_grad);
	layer->deltas = calloc((Batch*Out), sizeof *layer->deltas);

	layer->fwd = calloc(Batch*Out, sizeof *layer->fwd);
	layer->bwd = calloc(Batch*In, sizeof *layer->bwd);

	float (*weights)[In] = (void *)layer->weights;

	float sigma = sqrtf(2.0f / (float) In);
	for (int i=0; i<Out; ++i)
	for (int j=0; j<In; ++j) {

		float u1 = (float)rand()/RAND_MAX; //Uniform(0,1] random floats
		float u2 = (float)rand() / RAND_MAX;
		float radius = sqrt(-2 * log(u1));
		float theta = 2 * M_PI * u2;

		// First normal distributed value
		float z0 = radius * cos(theta) * sigma;
		// Second normal distributed value      
		float z1 = radius * sin(theta) * sigma;

		weights[i][j] = z0;  // normal distribution
	}

}

//L: number of layers
struct network *setup_network(int L, int *hidden_size, int batch_size) {
	struct network *nn = calloc(1, sizeof *nn);
	//nl -> number of hidden layers
	// input(784) -> hidden -> ... -> hiden -> output(10x)
	nn->L = L;
	nn->batch_size = batch_size;
	nn->layer = calloc(L, sizeof *nn->layer);

	for (int i=0, n=L; i<n; ++i) {
		setup_layer(&nn->layer[i],
	 	batch_size, hidden_size[i], hidden_size[i+1]);
		if (i) {  // connect layers
			nn->layer[i].a_list = nn->layer[i-1].fwd;
			nn->layer[i-1].deltas = nn->layer[i].bwd;
		}
	}

	return nn;
}


// loss function 
int loss(int batch, int b, float C[batch][b], const float A[batch][b], const float B[batch][b])
{
	int accuracy = 0;

	for (int i=0, n=batch; i<n; ++i) {
		int aimax = 0;
		int bimax = 0;
		float amax = A[i][0];
		float bmax = B[i][0];
		for (int j=0, jn=b; j<jn; ++j) {
			if (amax < A[i][j]) {
				aimax = j;
				amax = A[i][j];
			}
			if (bmax < B[i][j]) {
				bimax = j;
				bmax = B[i][j];
			}
		}
		accuracy += aimax == bimax;
		for (int j=0, jn=b; j<jn; ++j)
			C[i][j] = (A[i][j] - B[i][j]);
	}
	return accuracy;
}

void dot_fwd(float *act, float *weights, float *zout, int Batch, int In, int Out) {
	// act[Batch][In]
	// weigths[Out][In]
	// zout[Batch][Out]

	for(int ba=0; ba<Batch; ++ba)
		for (int k=0; k<Out; ++k) {
			zout[ba*Out+k] = 0.0; 
			for (int c=0; c<In; ++c) 
				zout[ba*Out+k] += act[ba*In+c] * weights[k*In+c];
		}
}



void dot_fwd_cblas(float *act, float *weights, float *zout, int Batch, int In, int Out) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // act[Batch][In] => A
    // weights[Out][In] => B,
    // zout[Batch][Out] => C

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 
                Batch, Out, In, 
                alpha, 
                act, In, 
                weights, In, 
                beta, 
                zout, Out);
}



void dot_fwd_cblas_cuda(float *act, float *weights, float *zout, int Batch, int In, int Out) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cudaDeviceProp deviceProp;
    int devID = 0;


    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 
                Batch, Out, In, 
                alpha, 
                act, In, 
                weights, In, 
                beta, 
                zout, Out);
}
    


void dot_bwd(float *deltas, float *weights, float *bwdout, int Batch, int In, int Out) {
	// del[Batch][Out]
	// wts[Out][In]
	// out[Batch][In]

    for (int ba=0; ba<Batch; ++ba)
		for (int c=0; c<In; ++c) {
			bwdout[ba*In+c] = 0.0;
			for (int k=0; k<Out; ++k) 
				bwdout[ba*In+c] += deltas[ba*Out+k] * weights[k*In+c];
			}
}

void dot_bwd_cblas(float *deltas, float *weights, float *bwdout, int Batch, int In, int Out) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // deltas[Batch][Out] => A
    // weights[Out][In] => B
    // bwdout[Batch][In] => C

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                Batch, In, Out, 
                alpha, 
                deltas, Out, 
                weights, In, 
                beta, 
                bwdout, In);
}

void dot_grd(float *act, float *del, float *grd, int Batch, int In, int Out) {
	// act[Batch][In]
	// del[Batch][Out]
	// grd[Out][In]

	for (int c=0; c<In; ++c) {
		for (int k=0; k<Out; ++k) {
			grd[k*In+c] = 0;
			for (int b=0; b<Batch; ++b) 
				grd[k*In+c] += act[b*In+c] * del[b*Out+k];
		}
	}
}

void dot_grd_cblas(float *act, float *del, float *grd, int Batch, int In, int Out) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // act[Batch][In] => A, 
    // del[Batch][Out] => B
    // grd[Out][In] => C

    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 
                Out, In, Batch, 
                alpha, 
                del, Out, 
                act, In, 
                beta, 
                grd, In);
}


void softmax(int batch, int b, float C[batch][b], float A[batch][b]) {
        for (int q=0, nq=batch; q<nq; ++q) {
                float max_value = A[q][0]; 
                float new_value[b];
		float exp_new_value[b];
                float sum = 0.0; 

		// Subtract lartest value for stability 
                for (int i=0, n=b; i<n; ++i)
                        max_value = MAX(max_value, A[q][i]);
                for (int i=0, n=b; i<n; ++i) 
                        new_value[i] = A[q][i] - max_value; 

		// Do softmax() 
                for (int i=0, n=b; i<n; ++i)
                        exp_new_value[i] = exp(new_value[i]);
                for (int i=0, n=b; i<n; ++i)
                        sum += exp_new_value[i];
                for (int i=0, n=b; i<n; ++i)
                        C[q][i] = exp_new_value[i]/sum;
        }
}           

void forward_pass(struct layer *ll, int lastlayer) {
	const int Batch = ll->batch;
	const int In = ll->in;
	const int Out = ll->out;
	float (*z_list)[Out] = (void *)ll->z_list;
	float (*fwd)[Out] = (void *)ll->fwd;

	float *act = (void *)ll->a_list;
	float *zout = (void *)ll->z_list;
	float *wts = (void *)ll->weights;

	dot_fwd_cblas((void *)ll->a_list, (void *)ll->weights, 
	              (void *)ll->z_list, Batch, In, Out);

	//dot_fwd((void *)ll->a_list, (void *)ll->weights, 
	//              (void *)ll->z_list, Batch, In, Out);

	for (int b=0; b<Batch; ++b)
	for (int k=0; k<Out; ++k)
		z_list[b][k] += ll->biases[k];

		
	// relu
	for (int b=0; b<Batch; ++b)
	for (int k=0; k<Out; ++k) {
		if ((z_list[b][k] >= 0.0) || lastlayer)
			fwd[b][k] = z_list[b][k];
		else 
			fwd[b][k] = 0.0;
	}       
}

void bwd_pass(struct layer *ll, int lastlayer) {

	const int Batch = ll->batch;
	const int In = ll->in;
	const int Out = ll->out;
	float (*deltas)[Out] = (void *)ll->deltas;
	float (*z_list)[Out] = (void *)ll->z_list;


	// relu derivative
	for (int b=0; b<Batch; ++b) {
		for (int k=0; k<Out; ++k) {
			if( (z_list[b][k]<=0) && (!lastlayer) )
				deltas[b][k] *= (0.0<z_list[b][k]);
		}
	}

	//dot_bwd((void *)ll->deltas, (void *)ll->weights, (void *)ll->bwd, Batch, In, Out);
	dot_bwd_cblas((void *)ll->deltas, (void *)ll->weights, (void *)ll->bwd, Batch, In, Out);

}

// upate the weights 
// sum gradients along the batch dimension
// wts <- eta*grd 
void weightupd(struct layer *ll, int num, int step) {
	const int Batch = ll->batch;
	const int In = ll->in;
	const int Out = ll->out;
	float *biases = ll->biases;

	float *act = (void *)ll->a_list;
	float *del = (void *)ll->deltas;
	float *grd = (void *)ll->w_grad;
	float *wts = (void *)ll->weights;


	//dot_grd(act, del, grd, Batch, In, Out);
	dot_grd_cblas(act, del, grd, Batch, In, Out);

	// weights
	for (int c=0; c<In; ++c) {
		for (int k=0; k<Out; ++k) {
			float g = grd[k*In+c];
			g /= Batch;
			wts[k*In+c] -= alpha*g;
		}
	}

	// biases
	for (int k=0; k<Out; ++k) {
		float g = 0;
		for (int ba=0; ba<Batch; ++ba)
			g += del[ba*Out+k];
		g /= Batch;
		biases[k] -= alpha*g;
	}

}

int train(struct network *nn, int nstep, int ntrain, void *ztrain, void *zlabel)
{
	const int L = nn->L;
    
	struct layer *layer = nn->layer;
	struct layer *first = &layer[0];
	struct layer *last  = &layer[L-1];

    const int nclass = last->out;
    const int batch = nn->batch_size;
	
	float (*images)[first->in] = (void *)first->a_list;
	float labels[batch][nclass];
	int tot_match = 0;

    const float (*const train)[first->in] = ztrain;
	const float (*const label)[10] = zlabel;

    for (int j=0, jn=nstep; j<jn; ++j) {

        // initialize input and labels
        for (int i=0, n=batch; i<n; ++i) {
			int z = rand()%ntrain;
			memcpy(images[i], train[z], sizeof images[i]);
			memcpy(labels[i], label[z], sizeof labels[i]);
		}

        /* forward pass */
		for (int i=0, n=L; i<n; ++i) {
			int lastlayer = (i==(L-1));  // to drop last relu
			forward_pass(&layer[i], lastlayer);
		}
		
		/* prediction */
		float p[batch][nclass];
        softmax(batch, nclass, p, (void *)last->fwd);

		/* loss */
	    int match = loss(batch,nclass, (void*)last->deltas, p, labels);
		tot_match += match;

		/* backward pass */
		for (int i=0, n=L; i<n; ++i) {
			int k = n-i-1;
			int lastlayer = (i==0);    // to drop last relu
			bwd_pass(&layer[k], lastlayer);
		}
	
		/* weight update */
		for (int i=0, n=L; i<n; ++i) {
			weightupd(&layer[i], i, j);
		}
	}
	return tot_match;
}


int test(struct network *nn, int nstep, int ntest, void *ztest, void *ztestlabel)
{
	int test_match = 0;

	const int L = nn->L;
    
	struct layer *layer = nn->layer;
	struct layer *first = &layer[0];
	struct layer *last  = &layer[L-1];

    const int nclass = last->out;
    const int batch = nn->batch_size;
	
	float (*testImages)[first->in] = (void *)first->a_list;
	float testLabels[batch][nclass];

    const float (*const test)[first->in] = ztest;
	const float (*const t_label)[10] = ztestlabel;


    for (int j=0, jn=nstep; j<jn; ++j) {

        // initialize input and labels
        for (int i=0, n=batch; i<n; ++i) {
			memcpy(testImages[i], test[i], sizeof testImages[i]);
			memcpy(testLabels[i], t_label[i], sizeof testLabels[i]);
		}

        /* forward pass */
		for (int i=0, n=L; i<n; ++i) {
			int lastlayer = (i==(L-1));  // to drop last relu
			forward_pass(&layer[i], lastlayer);
		}

		// predicion
		float p[batch][nclass];
        softmax(batch, nclass, p, (void *)last->fwd);

		// loss
	    int matching = loss(batch,nclass, (void*)last->deltas, p, t_label);
		test_match += matching;
	}

    
	return test_match;
}

void initalize_cuda() {
    devID = 0;
        error = cudaSetDevice(devID);

        if (error != cudaSuccess)
        {
            printf("cudaSetDevice returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }
    }
}

int main(const int argc, const char** argv){
    // read inputs
    int nl = atoi(argv[1]); // number of layers (exclusing in/out)
    int nh = atoi(argv[2]); // hidden dimension
    int ne = atoi(argv[3]); // number of epochs
    int nb = atoi(argv[4]); // batchsize
    alpha = atof(argv[5]); //learning rate

	//int seed = time(NULL); 
	int seed = 42;

    //initalize_cuda();

	// we only a subset of images (for quick tests)
    int imgs = 60000;  
	int test_imgs = 10000;  

	printf("layers = %d, batchsz = %d, seed = %d \n", 
		nl, nb, seed);

	srand(seed);

    int hidden_size[nl+1]; 
	hidden_size[0] = 28*28;
    for(int i=1; i<nl; ++i) hidden_size[i]=nh;
    hidden_size[nl] = 10; 

    int L = nl;

	struct network *nn = setup_network(L, hidden_size, nb);
    
    // Assuming input size for MNIST and output size for classification (10 digits)
    int inputSize = 784; // MNIST images are 28x28
    int outputSize = 10; // Digits 0-9

	// Load MNIST data
    int numImgs;
    float *images = readMNISTImages(TRAIN_IMG, &numImgs, &inputSize);
    int numLbls;
    float *labels = readMNISTLabels(TRAIN_LBL, &numLbls);

	int numTestImgs, testImgSize;
	float *testImages = readMNISTImages(TEST_IMG, &numTestImgs, &testImgSize);
	int numTestLbls;
	float *testLabels = readMNISTLabels(TEST_LBL, &numTestLbls);

	for (int i=0, n=ne; i<n; ++i) {
		clock_t start = clock();
		int match = train(nn, imgs/nb, numImgs, images, labels);
        float acc = ((float)match)/((float)imgs) * 100;
		printf("epoch %d: %d/%d correct, acc: %2.2f\n", 
		    i, match, imgs, acc);
		clock_t end = clock();
		float tElapsed = (float) (end - start) / CLOCKS_PER_SEC;
		if (i > 0) { // First iter is warm up
			totalTime += tElapsed;
		}
	}

	avgTime = totalTime / (float)(ne-1);
	float gr = (ne * 60000) / totalTime;

	printf("Grind Rate: %f\n", gr);
	printf("Total Time: %f\n", totalTime);

	printf("Average time per epoch: %f sec\n", avgTime);

	// Evaluate the network
	int test_match = test(nn, test_imgs/nb, numTestImgs, testImages, testLabels);
	float test_acc = ((float)test_match)/((float)numTestImgs) * 100;
	printf("%d/%d correct, acc: %2.2f\n", 
		test_match, test_imgs, test_acc);

	// Free resources
    free(images);
    free(labels);
	free(testImages);
	free(testLabels);

	return 0;

}
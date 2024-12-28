#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <stdint.h>
#include <assert.h>


//#include "mnist_loader.h"

#define ELTS(x) (sizeof(x)/sizeof*(x))
#define MAX(a,b) ((a)<(b)?(b):(a))
#define MIN(a,b) ((a)<(b)?(a):(b))

#define IMAGE_SIZE 28*28
#define NUM_TRAIN 60000

#define LABEL_SIZE 10
float alpha;

struct mnist {
        int ntrain;
        float (*image)[28*28];
        float (*label)[10];
};

struct mnist *mnist_load(void) ;

struct layer {
    int batch, in, out;
	float *a_list;  //activations size = (in*batch)
	float *z_list;  // output, size =(out * batch)

    float *fwd;   //to connect
	float *bwd;   // to connect

	float *biases;  // size = out
	float *weights; //matrix, size = in * out
	float *deltas;  //

};

struct network {
	int L;
	int batch_size;
	struct layer *layer; /* [L] */
};


struct mnist *mnist_load() {
        struct mnist *mnist = calloc(1, sizeof *mnist);
        FILE *f;

        struct {
                uint32_t magic;
                uint32_t count;
                uint32_t width;
                uint32_t height;
        } hdr_image = {0, };

        struct {
                uint32_t magic;
                uint32_t count;
        } hdr_label = {0, };

        f = fopen("../Data/train-images-idx3-ubyte", "rb");
        assert(f);
        fread(&hdr_image, sizeof hdr_image, 1, f);

        assert(hdr_image.magic == htonl(0x803));
        assert(ntohl(hdr_image.width) * ntohl(hdr_image.height) == 784);

        int count = ntohl(hdr_image.count);

        mnist->ntrain = count;
        mnist->image = calloc(count, sizeof *mnist->image);
        mnist->label = calloc(count, sizeof *mnist->label);

        for (int i=0, n=count; i<n; ++i) {
                unsigned char x[784];
                fread(x, sizeof x, 1, f);
                for (int j=0, jn=784; j<jn; ++j)
                        mnist->image[i][j] = (x[j] / 255.0) - 0.5;
        }
        fclose(f);

        f = fopen("../Data/train-labels-idx1-ubyte", "rb");
        assert(f);
        fread(&hdr_label, sizeof hdr_label, 1, f);

        assert(hdr_label.magic == htonl(0x801));
        assert(hdr_label.count == htonl(count));

        for (int i=0, n=count; i<n; ++i) {
                int y = fgetc(f);
                assert(0 <= y);
                assert(y <= 9);
                mnist->label[i][y] = 1.0;
        }

        return mnist;
}


void layer_create(struct layer *layer, int Batch, int In, int Out) {

    layer->batch = Batch;
    layer->in = In;
	layer->out = Out;
	
	layer->a_list = calloc((Batch*In), sizeof *layer->a_list);
    layer->z_list = calloc((Batch*Out), sizeof *layer->z_list);
    layer->biases = calloc((Batch*Out), sizeof *layer->biases);
    layer->weights = calloc((In*Out), sizeof *layer->weights);
	layer->deltas = calloc((Batch*Out), sizeof *layer->deltas);

    layer->fwd = calloc(Batch*Out, sizeof *layer->fwd);
	layer->bwd = calloc(Batch*In, sizeof *layer->bwd);

    float (*weights)[In] = (void *)layer->weights;
	//for (int i = 0; i < (In); i++) {
    //    layer->weights[i] = ((float)rand() / RAND_MAX) * 2.0 - 1.0; // Random values between -1 and 1
    //}
    for (int k=0; k<Out; ++k) {
		float s2 = 0;

		for (int c=0; c<In; ++c) {
			weights[k][c] = rand()/(RAND_MAX+1.0) - 0.5;
			s2 += weights[k][c] * weights[k][c];
		}
		s2 /= In;

		for (int c=0; c<In; ++c)
			weights[k][c] *= 1/sqrt(s2);

	}

}

//L: number of layers
struct network *network_create(int nl, int *nh, int batch_size) {
	struct network *nn = calloc(1, sizeof *nn);
    //nl -> number of hidden layers
    //784 -> 10x -> 10x -> output(10x)
	//nn->L = nl+1;
    nn->L = nl;
	nn->batch_size = batch_size;
	nn->layer = calloc(nn->L, sizeof *nn->layer);

    /*
    layer_create(&nn->layer[0], batch_size, IMAGE_SIZE, nh[1]);
    layer_create(&nn->layer[nn->L-1], batch_size, nh[nn->L-2], LABEL_SIZE);
    for (int i=1, n=nl; i<n; ++i) {
		layer_create(&nn->layer[i], batch_size, nh[i-1], nh[i+1]);
		nn->layer[i].a_list = nn->layer[i-1].fwd; //connect layers
		nn->layer[i-1].deltas = nn->layer[i].bwd;
	}
    */

   for (int i=0, n=nn->L; i<n; ++i) {
		layer_create(&nn->layer[i],
		  batch_size, nh[i], nh[i+1]);
		if (i) {
			nn->layer[i].a_list = nn->layer[i-1].fwd;
			nn->layer[i-1].deltas = nn->layer[i].bwd;
		}
	}

	return nn;
}


// loss function 
// compare argmax values of network prediction and labels 
// if match, increment count 
// return the difference (mean-squared error or L2)
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


void dot_fwd(float *act, float *wts, float *zout, int Batch, int In, int Out) {
        float (*a_list)[In] = (void *)act;
        float (*z_list)[Out] = (void *)zout;
        float (*weights)[In] = (void *)wts;

	for (int b=0; b<Batch; ++b)
		for (int k=0; k<Out; ++k) {
			z_list[b][k] = 0.0; 
			for (int c=0; c<In; ++c) 
				z_list[b][k] += a_list[b][c] * weights[k][c];
		}
}

void dot_bwd(float *deltas, float *weights, float *bwdout, int Batch, int In, int Out) {
        float (*d)[Out] = (void *)deltas;
        float (*w)[In] = (void *)weights;
	float (*bwd)[In] = (void *)bwdout;

        for (int b=0; b<Batch; ++b)
		for (int c=0; c<In; ++c) {
			bwd[b][c] = 0.0;
	                for (int k=0; k<Out; ++k) 
                                bwd[b][c] += d[b][k] * w[k][c];
                }
}

void forward_pass(struct layer *ll) {
    const int Batch = ll->batch;
	const int In = ll->in;
	const int Out = ll->out;
	float (*z_list)[Out] = (void *)ll->z_list;
        float (*fwd)[Out] = (void *)ll->fwd;


	dot_fwd((void *)ll->a_list, (void *)ll->weights, (void *)ll->z_list, Batch, In, Out);

    // sigmoid
	for (int b=0; b<Batch; ++b)
		for (int k=0; k<Out; ++k) {
			fwd[b][k] = (1.0 / (1.0 + exp(-z_list[b][k])));
		}

}

void bwd_pass(struct layer *ll) {
	const int Batch = ll->batch;
	const int In = ll->in;
	const int Out = ll->out;
	float (*deltas)[Out] = (void *)ll->deltas;

	// sigmoid derivative
	for (int b=0; b<Batch; ++b) {
		for (int k=0; k<Out; ++k) {
			float s = (1.0 / (1.0 + exp(-deltas[b][k])));
			deltas[b][k] *= s*(1.0-s);
		}
	}

	dot_bwd((void *)ll->deltas, (void *)ll->weights, (void *)ll->bwd, Batch, In, Out);
}

// upate the weights 
// sum gradients along the batch dimension
// wts <- eta*grd 
void weightupd(struct layer *ll, int num, int step) {
	const int Batch = ll->batch;
	const int In = ll->in;
	const int Out = ll->out;
	float (*a_list)[In] = (void *)ll->a_list;
	float (*deltas)[Out] = (void *)ll->deltas;
	float (*weights)[In] = (void *)ll->weights;


	for (int c=0; c<In; ++c) {
		for (int k=0; k<Out; ++k) {
			float grd = 0;
			for (int b=0; b<Batch; ++b) {
				grd += a_list[b][c] * deltas[b][k];
			}
			grd /= Batch;
			weights[k][c] -= alpha*grd;

		}
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
    const int Batch = batch;
	
	float (*images)[first->in] = (void *)first->a_list;
	float labels[batch][nclass];
	int accuracy = 0;

    const float (*const train)[first->in] = ztrain;
	const float (*const label)[10] = zlabel;

    printf("nstep:%d\n", nstep);


    for (int j=0, jn=nstep; j<jn; ++j) {

        // initialize input and labels
        for (int i=0, n=batch; i<n; ++i) {
			int z = rand()%ntrain;
			memcpy(images[i], train[z], sizeof images[i]);
			memcpy(labels[i], label[z], sizeof labels[i]);
		}

        /* forward pass */
		for (int i=0, n=L; i<n; ++i) {
			forward_pass(&layer[i]);
		}
		
		/* backward pass */
        int match = loss(batch, nclass, (void *)last->deltas, (void *)last->fwd, labels);
		
		accuracy += match;
		//printf("Step %d, accuracy: %d of %d\n", j, match, Batch);

		/* backward pass */
		for (int i=0, n=L; i<n; ++i) {
			int k = n-i-1;
			int bm = layer[k].batch * layer[k].out;
			bwd_pass(&layer[k]);
		}
	
		/* weight update */
		for (int i=0, n=L; i<n; ++i) {
			weightupd(&layer[i], i, j);
		}
	}
	return accuracy;
}


int main(const int argc, const char** argv){
    // read inputs
    int nl = atoi(argv[1]); // number of layers (exclusing in/out)
    int nh = atoi(argv[2]); // hidden dimension
    int ne = atoi(argv[3]); // number of epochs
    int nb = atoi(argv[4]); // batchsize
    float alpha = atof(argv[5]); //learning rate

	// load data
	//load_mnist();

    //nl+=1;

    int hidden_size[nl+1]; 
	hidden_size[0] = 28*28;
    for(int i=1; i<nl; ++i) hidden_size[i]=10;
    hidden_size[nl] = 10; 

    // we only a subset of images (for quick tests)
	int imgs = 5000; 
    //int imgs = 60000;  

    int batch_size = imgs/nb;
    printf("layers = %d, batch_size = %d, num_layers(full) = %d \n", nl, batch_size, nl+2);


    int nlayer = (sizeof(hidden_size)/sizeof*(hidden_size)) - 1;
    //printf("%d nlayer, %d nl\n",nlayer, nl);
	struct network *nn = network_create(nlayer, hidden_size, batch_size);
    
    struct mnist *mnist = mnist_load();
    int ntrain = mnist->ntrain;
	int a;

    int seed = 42;

    

    // initialize timer
	float totalTime = 0.0;
	float avgTime = 0.0;

	clock_t start = clock();

	for (int i=0, n=ne; i<n; ++i) {
		a = train(nn, nb, 
			ntrain, mnist->image, mnist->label);
		printf("epoch %d: %d/%d correct, batch_size:%d\n", i, a, imgs, batch_size);
        float perc = ((float)a)/((float)imgs) * 100;
        printf("accuracy:%f\n",perc);
		if (i % 10 == 9)
			alpha /= 2;  // learning rate schedule 

        clock_t end = clock();
        float tElapsed = (float) (end - start) / CLOCKS_PER_SEC;
        if (i > 0) { // First iter is warm up
            totalTime += tElapsed;
            avgTime = totalTime / (float)(i-1);
        }
	}

    printf("  Agerage time per epoch: %f sec\n", avgTime);

	return 0;

}
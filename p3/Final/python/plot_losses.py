import matplotlib.pyplot as plt

# Parse the loss values from the file
losses = []
with open('../extra/blas_loss.txt', 'r') as file:
    for line in file:
        if "Loss:" in line:
            _, loss_value = line.strip().split('Loss: ')
            losses.append(float(loss_value))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(losses, label='Training Loss')
plt.xlabel('Image Sample')
plt.ylabel('Loss')
plt.title('Training Loss Over Image Samples\n cBLAS, CPU')
plt.legend()
plt.grid(True)
plt.show()

plt.savefig('blas_loss.png')
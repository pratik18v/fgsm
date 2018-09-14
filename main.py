import pickle as pkl
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms

# Hyper-parameters 
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001
eps = 0.1

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
	def __init__(self, input_size, hidden_size, num_classes):
		super(NeuralNet, self).__init__()
		self.fc1 = nn.Linear(input_size, hidden_size) 
		self.relu = nn.ReLU()
		self.fc2 = nn.Linear(hidden_size, num_classes)  
	
	def forward(self, x):
		out = self.fc1(x)
		out = self.relu(out)
		out = self.fc2(out)
		return out

def train(device, train_loader):
	model = NeuralNet(input_size, hidden_size, num_classes).to(device)

	# Loss and optimizer
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

	# Train the model
	total_step = len(train_loader)
	for epoch in range(num_epochs):
		for i, (images, labels) in enumerate(train_loader):  
			# Move tensors to the configured device
			images = images.reshape(-1, 28*28).to(device)
			labels = labels.to(device)
			
			# Forward pass
			outputs = model(images)
			loss = criterion(outputs, labels)
			
			# Backward and optimize
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
			if (i+1) % 100 == 0:
				print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
							 .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

		# Save the model checkpoint
		torch.save(model.state_dict(), 'model/model-{}.ckpt'.format(epoch))

	return model

def test(device, test_loader):
	# Test the model
	# In test phase, we don't need to compute gradients (for memory efficiency)
	criterion = nn.CrossEntropyLoss()

	model = NeuralNet(input_size, hidden_size, num_classes).to(device)
	model.load_state_dict(torch.load("model/model-4.ckpt"))
		
	correct = 0
	adv_correct = 0
	misclassified = 0
	total = 0
	noises = []
	y_preds = []
	y_preds_adv = []
	for images, labels in test_loader:
		images = Variable(images.reshape(-1, 28*28).to(device), requires_grad=True)
		labels = Variable(labels.to(device))

		outputs = model(images)
		loss = criterion(outputs, labels)
		loss.backward()

		#Add perturbation
		grad = torch.sign(images.grad.data)
		imgs_adv = torch.clamp(images.data + eps * grad, 0, 1)

		adv_outputs = model(Variable(imgs_adv))

		_, predicted = torch.max(outputs.data, 1)
		_, adv_preds = torch.max(adv_outputs.data, 1)

		total += labels.size(0)
		correct += (predicted == labels).sum().item()
		adv_correct += (adv_preds == labels).sum().item()
		misclassified += (predicted != adv_preds).sum().item()

		noises.extend((images - imgs_adv).data.numpy())
		y_preds.extend(predicted.data.numpy())
		y_preds_adv.extend(adv_preds.data.numpy())

	print('Accuracy of the network w/o adversarial attack on the 10000 test images: {} %'.format(100 * correct / total))
	print('Accuracy of the network with adversarial attack on the 10000 test images: {} %'.format(100 * adv_correct / total))
	print('Number of misclassified examples (as compared to clean predictions): {}/{}'.format(misclassified, total))

	with open("mnist_fgsm.pkl","w") as f: 
			data_dict = {
							"noises" : noises,
							"y_preds" : y_preds,
							"y_preds_adv" : y_preds_adv
							}    
			pkl.dump(data_dict, f)

def main(flag):
	# Device configuration
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		# MNIST dataset 
	train_dataset = torchvision.datasets.MNIST(root='./data', 
																						 train=True, 
																						 transform=transforms.ToTensor(),  
																						 download=True)

	test_dataset = torchvision.datasets.MNIST(root='./data', 
																						train=False, 
																						transform=transforms.ToTensor())

	# Data loader
	train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
																						 batch_size=batch_size, 
																						 shuffle=True)

	test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
																						batch_size=batch_size, 
																						shuffle=False)

	if flag	== "train":
		model = train(device, train_loader)
	elif flag == "test":
		test(device, test_loader)

if __name__ == "__main__":
	main("test")

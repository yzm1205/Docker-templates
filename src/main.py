import torch
import torchvision
from util import get_accuracy
from model import Model
import torchvision.transforms as transforms

BATCH_SIZE = 32

## transformations
transform = transforms.Compose(
    [transforms.ToTensor()])
    
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")    

 ## download and load testing dataset
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                    download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                        shuffle=False, num_workers=0)
    
def test(model,testloader,BATCH_SIZE):
    # test the model
    with torch.no_grad():
        test_acc = 0
        for i,(image,label) in enumerate(testloader):
            image = image.to(device)
            label = label.to(device)
            output = model(image)
            test_acc += get_accuracy(output,label,BATCH_SIZE)
    print('Test Accuracy: %.2f'%(round(test_acc/i,3)))
    return test_acc/i


    
    

if __name__ == '__main__':
    # freeze_support()
    # train = train(model,trainloader,param_dict,save=True)
    model = Model().to(device)
    model.load_state_dict(torch.load('./Model_saved/model.pth'))
    model.eval()
    test = test(model,testloader,BATCH_SIZE)
    print("Done.")
    
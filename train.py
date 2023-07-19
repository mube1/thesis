

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import argparse

import model as m
import eval
import dataloader

BATCH_SIZE = 32
CHECKPOINT_EVERY = 50
EPOCHS = 100
LEARNING_RATE = 1e-3
SAMPLE_SIZE = 100000
L2_REGULARIZATION_STRENGTH = 0
EPSILON = 0.001
DATASET=10
STUDENT=8
TEACHER=101
PRUNING_RATE=0
TEMPERATURE=1
PRUNING_MODE='forward'
DATA_LOCATION='data'

# optimizers=[
#   torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.00001, momentum=0.9),
#   torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.00001, momentum=0.9),
#   torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.00001, momentum=0.9),
  

#   ]


def get_arguments():

    parser = argparse.ArgumentParser(description='Trainer')

    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='How many wav files to process at once. Default: ' + str(BATCH_SIZE) + '.')
  
    parser.add_argument('--logdir_root', type=str, default=None,
                        help='Root directory to place the logging '
                        'output and generated model. These are stored '
                        'under the dated subdirectory of --logdir_root. '
                        'Cannot use with --logdir.')
    
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help='Number of epochs. Default: ' + str(EPOCHS) + '.')
    
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help='Learning rate for training. Default: ' + str(LEARNING_RATE) + '.')
  
 
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam','sgd'],
                        help='Select the optimizer specified by this option. Default: adam.')
    
    parser.add_argument('--scheduler', type=str, default=None, choices=['CosineAnnealingLR','OneCycleLR'],
                        help='Select the scheduler specified by this option. Default: None.')
   
    
    parser.add_argument('--dataset', type=int, default=DATASET,
                        help='Cifar dataset to be used. Default : Cifar :' + str(DATASET))
    
    parser.add_argument('--student', type=int, default=STUDENT,
                        help='Student model type from Resnet family. Default : ' + str(STUDENT))
    
    parser.add_argument('--teacher', type=int, default=TEACHER,
                        help='teacher model type from Resnet family. Default : ' + str(TEACHER))
    
    parser.add_argument('--loss', type=str, default=None,
                        help='Type of loss function. Default : CrossEntropy()')
    
    parser.add_argument('--pruning_rate', type=float, default=PRUNING_RATE,
                        help='Pruning rate S. When S =0, it the algorithm proposed falls to Hintons KD. Default :'+str(0))
    
    parser.add_argument('--pruning_mode', type=str, default=PRUNING_MODE,
                        help='Pruning mode, which part to remove: forward means remove smaller, reverse means remove bigger.Default : '+PRUNING_MODE)
    
    parser.add_argument('--temperature', type=int, default=TEMPERATURE,
                        help='Temperature of KD. Default :'+str(TEMPERATURE))
    
    parser.add_argument('--data_download', type=bool, default=True, help='To download or use local data.' )
    
    parser.add_argument('--data_location', type=str, default=DATA_LOCATION, help='Data location downloaded or otherwise.' )
    
    
    return parser.parse_args()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def logit_pruning(a,rate=.5, p_type='forward'):
    #     magnitude pruning....takes percentage of a to be used
    t=max(int(a.shape[1]*(1-rate)),1)
    v=a.sort(1).indices
    if p_type=='forward':
        s=sum([a*F.one_hot(v[:,-(i+1)],a.shape[1]) for i in range(t)])
    else:
        s=sum([a*F.one_hot(v[:,i],a.shape[1]) for i in range(t)])
    return s


def train(student, 
          teacher, 
          epochs,
          optimizer,
          pruning_rate,
          pruning_mode,
          criterion,
          train_loader,
          test_loader,
          scheduler,
          temperature=10.0):
    
    student.to(device)

    teacher.eval()
    teacher.to(device)

    kl_div_loss = nn.KLDivLoss(log_target=True)
    soft_targets_weight: float = 100.
    temperature: float = temperature*temperature
    label_loss_weight: float = 0.5


 # Training
    student.train()

    for epoch in tqdm(range(epochs)):

        running_loss = 0
        running_corrects = 0

        for inputs, labels in train_loader:
        

            inputs = inputs.to(device)
            labels = labels.to(device)


            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = student(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            if pruning_rate>0 and pruning_rate <1:
                logits=teacher(inputs)
                soft_prob = nn.functional.softmax(outputs / temperature, dim=-1)
                soft_targets = nn.functional.softmax( logits/temperature, dim=-1)
                soft_targets=logit_pruning(soft_targets,rate=pruning_rate,p_type=pruning_mode)
                soft_targets=(soft_targets.T/soft_targets.sum(1)).T  #                 re normalize


                soft_targets_loss = kl_div_loss(soft_prob, soft_targets)
                loss = soft_targets_weight * soft_targets_loss + label_loss_weight * loss

            elif pruning_rate == 0:
                logits=teacher(inputs)
                soft_prob = nn.functional.softmax(outputs / temperature, dim=-1)
                soft_targets = nn.functional.softmax( logits/temperature, dim=-1)
                
                soft_targets_loss = kl_div_loss(soft_prob, soft_targets)
                loss = soft_targets_weight * soft_targets_loss + label_loss_weight * loss


            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()


            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = running_corrects / len(train_loader.dataset)

        eval_loss, eval_accuracy = eval.evaluate_model(model=student, test_loader=test_loader, device=device, criterion=criterion)


def get_optimizer(model,opt,lr,wd=0.00001,m=.9):
    if opt=='adam':
        return torch.optim.Adam(model.parameters(), lr, wd, momentum=m)
    else:
        return torch.optim.Adam(model.parameters(), lr, wd, momentum=m)


def main():

#   teacher=torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet56", pretrained=True)
  teacher=101
  
  
  args = get_arguments()


  train_loader, test_loader = dataloader.prepare_dataloader(args.data_location,cifar=100,train_batch_size=args.batch_size,eval_batch_size=args.batch_size,download=args.data_download)
  
  student=m.get_models(args.student,args.dataset)
  optimizer = get_optimizer(student,args.optimizer,args.learning_rate)

  train(
        args.student, # 
        args.teacher,  # 
        # args.learning_rate,
        args.epochs, # 
        optimizer, # 
        args.pruning_rate,
        args.pruning_mode,
        args.loss, # 
        train_loader, # 
        test_loader, # 
        args.scheduler, # 
        args.temperature

         )
  
  


if __name__ == '__main__':
    main()

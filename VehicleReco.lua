
require "torch"
require "nn"
require "math"

-- global variables
DataMean = 0
DataStd  = 0
hand     = 1
nonhand  = 2
w        = 32
h        = 32

torch.manualSeed(1)                  
s = torch.load('RngState.dat')
torch.setRNGState(s)

function create_network(nb_outputs)

   local ann = nn.Sequential();  -- make a multi-layer structure
   
   -- input h*w*1 32*32*1
   ann:add(nn.SpatialConvolution(1,20,3,3))    --  30x30x20
   ann:add(nn.SpatialMaxPooling(2,2,2,2))      --  15x15x20

   ann:add(nn.SpatialConvolution(20,20,4,4))   --  12x12x20x20
   ann:add(nn.SpatialMaxPooling(2,2,2,2))      --  6x6x20
   
   ann:add(nn.Reshape( 6*6*20 ))  -- 6250
   ann:add(nn.ReLU())
   ann:add(nn.Dropout(0.5))
   ann:add(nn.Linear(  6*6*20 , 100 ))
   ann:add(nn.ReLU())
   ann:add(nn.Dropout(0.5))
   ann:add(nn.Linear(  100  , 100 ))
   ann:add(nn.ReLU())
   ann:add(nn.Dropout(0.5))
   ann:add(nn.Linear( 100,nb_outputs ))
   ann:add(nn.LogSoftMax())
   
   return ann
end

-- train a Neural Netowrk
function train_network(network, training_dataset,testing_dataset, classes, classes_names)
   local criterion = nn.ClassNLLCriterion()

   trainer = nn.StochasticGradient(network, criterion)
   trainer.learningRate = 0.005
   trainer.maxIteration = 1      --8  epochs of training.
   maxtIteration = 30
   for i =1,maxtIteration do
      local _start = os.time()
      print('---------------------------------------[ ',i,' ]')
      trainer:train(training_dataset)
      predictedLabels = test_predictor(network, testing_dataset, classes, classes_names)  
      local _end = os.time()
      print('takes',_end-_start,'s')
   end
   print('---------------------------------------')
end

function test_predictor(predictor, test_dataset, classes, classes_names)

        local mistakes = 0
        local tested_samples = 0
        local predictedLabels = {};
        local FP = 0
        local FN = 0

        for i=1,test_dataset:size() do

               local input    = torch.Tensor(1,h,w):zero()
               local class_id = test_dataset[i][2]
               input[1] = test_dataset[i][1]

               local responses_per_class  =  predictor:forward(input) 
               local probabilites_per_class = torch.exp(responses_per_class)
               local probability, prediction = torch.max(probabilites_per_class, 1) 

               predictedLabels[i] =  prediction[1];
               if class_id==hand    and prediction[1]==nonhand then
                    FN = FN+1
               end
               if class_id==nonhand and prediction[1]==hand then
                    FP = FP+1
               end
               if class_id~=prediction[1] then
                      mistakes = mistakes + 1
                      local label = classes_names[ classes[class_id] ]
                      local predicted_label = classes_names[ classes[prediction[1] ] ]
                      --print(i , label , predicted_label )
               end

               tested_samples = tested_samples + 1
        end

        local test_err = mistakes/tested_samples
        print ( "Test error " .. test_err .. " ( " .. mistakes .. " out of " .. tested_samples .. " )")
        print ( "False positive = ",FP)
        print ( "False negative = ",FN)
        return predictedLabels
end

function wait(seconds)
        local _start = os.time()
        local _end = _start+seconds
        while (_end ~= os.time()) do
        end
end

-- main routine
function main()
        --------------------------------------------------------------------------------------------
        local training_dataset, testing_dataset, classes, classes_names = dofile('loadDataset.lua')
        network = create_network(#classes)
        train_network(network, training_dataset,testing_dataset, classes, classes_names)
        torch.save('network.dat',network)

        predictedLabels = test_predictor(network, testing_dataset, classes, classes_names)
        --------------------------------------------------------------------------------------------
        --[[
        DataMean = training_dataset[1][{ {}, {}, {}  }]:mean() -- mean estimation
        print('Mean: ' .. DataMean)
        --training_dataset[1][{ {}, {}, {}  }]:add(-DataMean)    -- mean subtraction
        
        DataStd = training_dataset[1][{ {}, {}, {}  }]:std() -- std estimation
        print('Standard Deviation: ' .. DataStd)
        --training_dataset[1][{ {}, {}, {}  }]:div(DataStd) -- std scaling
        --[[
        for i=1,10 do     --testing_dataset:size()
            itorch.image( image.scale(testing_dataset[i][1],400,'simple'))
            print('label      : ',classes_names[ testing_dataset[i][2] ])
            print('prediction : ',classes_names[ predict(network,testing_dataset[i][1])   ])
            wait(1)
        end
        ]]--
        --------------------------------------------------------------------------------------------
        
      
end

main()



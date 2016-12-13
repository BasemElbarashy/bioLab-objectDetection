require "torch"
require "image"
require "math"

s = torch.load('RngState.dat')
torch.setRNGState(s)

hand    = 1
nonhand = 2
classes 	  = { hand , nonhand } -- indices in torch/lua start at 1, not at zero
classes_names = {'hand','nonhand'}

w    = 32
h    = 32
trainToAllRatio = 0.8

posFolder = 'hand_dataset/training_dataset/pos/'
negFolder = 'hand_dataset/training_dataset/neg/'


function shuffle(array)
    local counter = #array
    function swap(array, index1, index2)  array[index1], array[index2] = array[index2], array[index1] end

    while counter > 1 do
        local index = torch.random(counter)
        swap(array, index, counter)
        counter = counter - 1
    end
end


function load_data_from_disk()
	local dataset={}
	local  trainset, testset = {},{}
	--local labels = torch.Tensor(nNegAndPos,1):zero()
	local groupSize   = torch.Tensor{    2865         ,    2865     }
	local groupClass  = torch.Tensor{    hand         ,  nonhand    }
	local groupFormat =             {	'pos_%d.png'  ,'neg_%d.png' }
	local groupFolder =             { 	posFolder     , negFolder	}
	local grouptStart =             {      0          ,     0       }
			   
	local datasetIdx  = 1;

	for groupIdx = 1,groupSize:size(1) do

	   for i = datasetIdx, datasetIdx + groupSize[groupIdx] -1 do
	      local filename = string.format( groupFormat[groupIdx] ,i-datasetIdx+grouptStart[groupIdx])
	      local input = image.load(groupFolder[groupIdx] .. filename)      -- images_set is global

	      if(input:size(1) == 3) then
	      		input = (input[{{1},{},{}}] + input[{{2},{},{}}] + input[{{3},{},{}}] ) / 3
	      end
	      --[[
	      if(groupIdx == 3) then
	      		input = input[{{},{33,96},{}}]
	      end
		  ]]--
	      input = image.scale(input,w,h)	      

	      dataset[i] = {input, groupClass[groupIdx]}-- class 2
	   end
	   
	   datasetIdx = datasetIdx + groupSize[groupIdx];

	end
	datasetLength = datasetIdx - 1

	shuffle(dataset)

	nTrain = trainToAllRatio * datasetLength
	nTest  = datasetLength   - nTrain
	
	for i = 1,nTrain do
		trainset[i] = dataset[i]
	end

	j=0;
	for i = 1,nTest do
		testset[i]  = dataset[nTrain+i]
		if testset[i][2] == 1 then		j=j+1 end
	end
	print(j,'/',nTest)

	function trainset:size()  return nTrain end
	function testset:size()   return nTest end

	return trainset, testset 
end



local  trainset, testset = load_data_from_disk()

return trainset, testset, classes, classes_names

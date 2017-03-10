#!/usr/bin/env th

require 'image';
require 'nn';
require 'paths';
require 'lfs';
require 'cutorch';
require 'cunn';
csv = require 'csv';

local train_dir = "./train"
local val_dir = "./val"

local train_size = 0;
print("Counting training images...");
for img_name in lfs.dir(train_dir) do
	path = train_dir .. '/' .. img_name;
	
	if lfs.attributes(path, "mode") == "file" then
		-- print("Found image " .. path);
		train_size = train_size + 1;
	elseif lfs.attributes(path, "mode") == "directory" then
		print("Skipping subdirectory " .. path);
	end
end

print("Found ", train_size, " training images.");
if train_size == 0 then
	io.stderr:write("No images found in training images directory!");
	os.exit(2);
end

local val_size = 0;
print("Counting validation images...");
for img_name in lfs.dir(train_dir) do
	path = val_dir .. '/' .. img_name;
	
	if lfs.attributes(path, "mode") == "file" then
		-- print("Found image " .. path);
		val_size = val_size + 1;
	elseif lfs.attributes(path, "mode") == "directory" then
		print("Skipping subdirectory " .. path);
	end
end

print("Found ", val_size, " validation images.");
if val_size == 0 then
	io.stderr:write("No images found in validation images directory!");
	os.exit(2);
end

local aug_dir = "./aug_train"

local aug_size = 0;
-- print("Counting augmented images...");
-- for img_name in lfs.dir(aug_dir) do
-- 	path = aug_dir .. '/' .. img_name;
	
-- 	if lfs.attributes(path, "mode") == "file" then
-- 		-- print("Found image " .. path);
-- 		aug_size = aug_size + 1;
-- 	elseif lfs.attributes(path, "mode") == "directory" then
-- 		print("Skipping subdirectory " .. path);
-- 	end
-- end

-- print("Found ", aug_size, " augmented images.");

local input_channels = 3;
local input_height = 128;
local input_width = 128;

local num_classes = 8;

local dataset = {};
function dataset:size() return (train_size + aug_size); end
setmetatable(dataset,
	{__index = function(self, index)
		local input = self.data[index];
		local class = self.label[index];
		local example = {input, class};
		return example;
	end}
);

dataset.data = torch.CudaTensor(train_size+aug_size, input_channels, input_height, input_width);
dataset.label = torch.CudaTensor(train_size+aug_size);

local img_to_label = {}
local ind_to_img = {}
local label_file = csv.open("train.csv")
local aug_file = csv.open("aug_train.csv");

for entry in label_file:lines() do
	img_to_label[entry[1]] = entry[2];
end
for entry in aug_file:lines() do
	if (entry[1] ~= "Id") then
		img_to_label[tostring(tonumber(entry[1]) + train_size)] = entry[2];
	end
end

local i = 1;

print("Loading training images...");
for img_name in lfs.dir(train_dir) do
	path = train_dir .. '/' .. img_name;
	if lfs.attributes(path, "mode") == "file" then
		-- print("Loading image " .. path);
		local img = image.load(path, input_channels, 'float');
		local img_tensor = img:view(input_channels, input_height, input_width);

		dataset.data[i] = img_tensor:cuda();
		local target = tonumber(img_to_label[tostring(tonumber(img_name:sub(1,5)))]);
		dataset.label[i] = target;
		ind_to_img[i] = tonumber(img_name:sub(1,5));
		i = i + 1;
	end
end
-- for img_name in lfs.dir(aug_dir) do
-- 	path = aug_dir .. '/' .. img_name;
-- 	if lfs.attributes(path, "mode") == "file" then
-- 		-- print("Loading image " .. path);
-- 		local img = image.load(path, input_channels, 'float');
-- 		local img_tensor = img:view(input_channels, input_height, input_width);

-- 		dataset.data[i] = img_tensor:cuda();
-- 		local target = tonumber(img_to_label[tostring(tonumber(img_name:sub(1,5)) + train_size)]);
-- 		dataset.label[i] = target;
-- 		i = i + 1;
-- 	end
-- end

-- Data preprocessing
local mean_image = torch.mean(dataset.data, 1);

-- Feature scaling
for k =1,train_size do
	dataset.data[k] = dataset.data[k] - mean_image;
end

local net = nn.Sequential();

net:add(nn.Reshape(input_channels * input_height * input_width));
net:add(nn.Linear(input_channels * input_height * input_width, num_classes));

-- net:add(nn.SpatialConvolution(3, 8, 5, 5, 1, 1));
-- net:add(nn.Sigmoid());
-- net:add(nn.SpatialMaxPooling(4, 4, 2, 2));
-- net:add(nn.Reshape(8 * 61 * 61));
-- net:add(nn.Linear(8 * 61 * 61, num_classes));

-- net:add(nn.SpatialConvolution(3, 24, 8, 8, 4, 4));
-- net:add(nn.ReLU());
-- net:add(nn.SpatialConvolution(24, 128, 5, 5, 2, 2));
-- net:add(nn.ReLU());
-- net:add(nn.SpatialConvolution(128, 192, 3, 3));
-- net:add(nn.ReLU());
-- net:add(nn.SpatialConvolution(192, 192, 3, 3));
-- net:add(nn.ReLU());
-- net:add(nn.SpatialConvolution(192, 128, 3, 3));
-- net:add(nn.ReLU());
-- net:add(nn.Reshape(128*8*8));
-- net:add(nn.Linear(128*8*8, 4096));
-- net:add(nn.ReLU());
-- net:add(nn.Dropout(0.5));
-- net:add(nn.Linear(4096, 1024));
-- net:add(nn.ReLU());
-- net:add(nn.Dropout(0.5));
-- net:add(nn.Linear(1024, 512));
-- net:add(nn.ReLU());
-- net:add(nn.Dropout(0.5));
-- net:add(nn.Linear(512, 256));
-- net:add(nn.ReLU());
-- net:add(nn.Dropout(0.5));
-- net:add(nn.Linear(256, num_classes));

print('Network:\n' .. net:__tostring() .. '\n\n');

local loss = nn.CrossEntropyCriterion();

local trainer = nn.StochasticGradient(net:cuda(), loss:cuda());
trainer.learningRate = 0.01;
trainer.learningRateDecay = 0.001;
trainer.maxIteration = 100;
trainer.shuffleIndices = true;

trainer:train(dataset);

net:add(nn.SoftMax());

print("Scoring training images...");
train_out = io.open("train_out.csv", "w");
train_out:write("Id,Label,Usage\n")
local corr = 0;
local cudanet = net:cuda();
for k = 1,train_size do
	local input = dataset.data[k];
	local target = dataset.label[k];
	local maxs, inds = torch.max(cudanet:forward(input), 1);

	train_out:write(tostring(ind_to_img[k]) .. "," .. tostring(inds[1]) .. ",Public\n");
	
	if inds[1] == target then
		corr = corr + 1;
	end
end
io.close(train_out);
print("(Top-1) Train accuracy: " .. tostring(corr/dataset:size()) .. " " .. tostring(corr) .. "/" .. tostring(dataset:size()));

val_out = io.open("val.csv", "w");
print("Scoring validation images...");
val_out:write("Id,Prediction\n");

for img_name in lfs.dir(val_dir) do
	path = val_dir .. '/' .. img_name;
	if lfs.attributes(path, "mode") == "file" then
 		local img = image.load(path, input_channels, 'float');
 		local img_tensor = img:view(input_channels, input_height, input_width);
 		local maxs, inds = torch.max(cudanet:forward(img_tensor:cuda() - mean_image), 1);
 		val_out:write(tostring(tonumber(img_name:sub(1,5))) .. "," .. tostring(inds[1]) .. "\n");
	end
end

for l = val_size+1,2970 do
	val_out:write(tostring(l) .. "," .. tostring(0) .. "\n");
end

io.close(val_out);

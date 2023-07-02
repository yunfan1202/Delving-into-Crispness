clear;clc;
% addpath(genpath('../crisp_edge'));

dir_name= 'crispness_example/raw_prediction';
%dir_name= '../datasets/crispness_example/refine_prediction';

list_ed = dir(fullfile(dir_name,'*.png'));

n_res = length(list_ed);
is_edge_white = false; % for the nms the edges have to be in white and non-edges in black

disp('==> Please wait NMS is been applyed to your predictions...');

total_crisp = 0;
min_crisp = 1;
max_crisp = 0;
tic
for i=1:n_res
    tmp_edge = imread(fullfile(dir_name,list_ed(i).name));
       
    if length(size(tmp_edge))>2
        tmp_edge = rgb2gray(tmp_edge);
    end
    if ~is_edge_white
        tmp_edge = 1-single(tmp_edge)/255; % image incomplement
    else
        tmp_edge = single(tmp_edge)/255; 
    end
   
    tmp= tmp_edge;
    edg=tmp_edge;
    
    [Ox, Oy] = gradient2(convTri(tmp_edge,4));
    [Oxx, ~] = gradient2(Ox); [Oxy,Oyy] = gradient2(Oy);

    O = mod(atan(Oyy.*sign(-Oxy)./(Oxx+1e-5)),pi);
    tmp_edge = edgesNmsMex(tmp_edge,O,2,5,1.01,8); % the original is 4   
    crispness=sum(sum(tmp_edge)) / sum(sum(edg));
    % disp(crispness);
    if crispness > max_crisp
        max_crisp = crispness;
    end
    if crispness < min_crisp
        min_crisp = crispness;
    end
    
    total_crisp = total_crisp + crispness;
end

toc

avg_crisp = roundn(total_crisp / n_res,-3);
min_crisp = roundn(min_crisp,-3);
max_crisp = roundn(max_crisp,-3);
res_inf = ['avg_crisp=' num2str(avg_crisp) ' min_crisp=' num2str(min_crisp) ' max_crisp=' num2str(max_crisp)];
disp(res_inf);

disp('NMS process finished...');

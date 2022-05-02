clc;clear;close all;
%% Loading Data

path='.\dataset\B\*.txt' ;  
files=dir(path);

for i = 1:length(files)
    fn = [path(1:end-5) files(i,1).name];
    Input_test=load(fn);
    
    %%Denoise
     wname='db10'; %WaveName
    [C,L]=wavedec(Input_test,4,wname);   
    [THR,SORH,KeepApp]=ddencmp('den','wv',Input_test);  
    Input_test= wdencmp('gbl',C,L,wname,4,THR,SORH,KeepApp);  
    
    
WaveletFunction='db8'; %OR 'sym8' Symlet8
    [C,L]=wavedec(Input_test,8,WaveletFunction);
    %% Calculate The Coefficient Vector
    
    %%%CoefficientDtail=DetailCoefficient(C,L,level);
    
    cD1=detcoef(C,L,1);  %Noisy
    cD2=detcoef(C,L,2);  %Noisy
    cD3=detcoef(C,L,3);  %Noisy
    cD4=detcoef(C,L,4);  %Noisy
    cD5=detcoef(C,L,5);  %Gama
    cD6=detcoef(C,L,6);  %Beta
    cD7=detcoef(C,L,7);  %Alpha
    cD8=detcoef(C,L,8);  %Teta
    cA8=appcoef(C,L,WaveletFunction,8);  %Delta   %CoefficientApproximation=ApproximationCoefficient(C,L,level)
    
    
    %%%%Calculate The Details Vector
        
    D1=wrcoef('d',C,L,WaveletFunction,1);    %Noisy
    D2=wrcoef('d',C,L,WaveletFunction,2);    %Noisy
    D3=wrcoef('d',C,L,WaveletFunction,3);    %Noisy
    D4=wrcoef('d',C,L,WaveletFunction,4);    %Noisy
    D5=wrcoef('d',C,L,WaveletFunction,5);    %Gama
    D6=wrcoef('d',C,L,WaveletFunction,6);    %Beta
    D7=wrcoef('d',C,L,WaveletFunction,7);    %Alpha
    D8=wrcoef('d',C,L,WaveletFunction,8);    %Teta
    A8=wrcoef('d',C,L,WaveletFunction,8);    %Delta
    
    %Make Features Vector
    f1=FMD(D5);
    f2=FMD(D6);
    f3=FMD(D7);
    f4=FMN(D5);
    f5=FMN(D6);
    f6=FMN(D7);
    f7=FR(D5);
    f8=FR(D6);
    f9=FR(D7);
    f10=WL(D5);
    f11=WL(D6);
    f12=WL(D7);
    
    Feature_Normal(i,:)=[f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12];
    
end
    
% save('Feautre_Normal.mat','Feature_Normal');

%% Abnormal Data %Epilepsy


path='.\dataset\E\*.txt' ;  
files=dir(path);

for i = 1:length(files)
    fn = [path(1:end-5) files(i,1).name];
    Input_test=load(fn);

       %%Denoise
    wname='db10'; %WaveName
    [C,L]=wavedec(Input_test,4,wname);   
   [THR,SORH,KeepApp]=ddencmp('den','wv',Input_test); 
    Input_test= wdencmp('gbl',C,L,wname,4,THR,SORH,KeepApp);  
    
    
WaveletFunction='db8'; %OR 'sym8' Symlet8
    [C,L]=wavedec(Input_test,8,WaveletFunction);
    %% Calculate The Coefficient Vector
    
    %%%CoefficientDtail=DetailCoefficient(C,L,level);
    
    cD1=detcoef(C,L,1);  %Noisy
    cD2=detcoef(C,L,2);  %Noisy
    cD3=detcoef(C,L,3);  %Noisy
    cD4=detcoef(C,L,4);  %Noisy
    cD5=detcoef(C,L,5);  %Gama
    cD6=detcoef(C,L,6);  %Beta
    cD7=detcoef(C,L,7);  %Alpha
    cD8=detcoef(C,L,8);  %Teta
    cA8=appcoef(C,L,WaveletFunction,8);  %Delta  
    
    %%%%Calculate The Details Vector
    
    
    D1=wrcoef('d',C,L,WaveletFunction,1);    %Noisy
    D2=wrcoef('d',C,L,WaveletFunction,2);    %Noisy
    D3=wrcoef('d',C,L,WaveletFunction,3);    %Noisy
    D4=wrcoef('d',C,L,WaveletFunction,4);    %Noisy
    D5=wrcoef('d',C,L,WaveletFunction,5);    %Gama
    D6=wrcoef('d',C,L,WaveletFunction,6);    %Beta
    D7=wrcoef('d',C,L,WaveletFunction,7);    %Alpha
    D8=wrcoef('d',C,L,WaveletFunction,8);    %Teta
    A8=wrcoef('d',C,L,WaveletFunction,8);    %Delta
    
    %Make Features Vector
    f1=FMD(D5);
    f2=FMD(D6);
    f3=FMD(D7);
    f4=FMN(D5);
    f5=FMN(D6);
    f6=FMN(D7);
    f7=FR(D5);
    f8=FR(D6);
    f9=FR(D7);
    f10=WL(D5);
    f11=WL(D6);
    f12=WL(D7);
    
    Feature_Epilepsy(i,:)=[f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12];
    
end
    
%% Creating Input & Output

Input=[Feature_Normal;Feature_Epilepsy];
Output=[zeros(length(Feature_Normal),1);ones(length(Feature_Epilepsy),1)];

%% Train SVM with 5-fold cross validation

k=5;

% % % % Create a cvpartition object that defined the folds
c1 = cvpartition(Output,'Kfold',k);

for i=1:k
% % % % % Create a training set
Input_test = Input(training(c1,i),:);
Label_test = Output(training(c1,i),:);
% % % % % % test set
Input_train=Input(test(c1,i),:);
Laebl_train=Output(test(c1,i),:);

svmmdl=fitcsvm(Input_test,Label_test,'Standardize',true,'KernelFunction','RBF',...
    'KernelScale','auto');

output=predict(svmmdl,Input_train);
% figure()
% plotconfusion(v',output')
% title('SVM classifier Confusion Matrix')


[c,cm,ind,per] = confusion(Laebl_train',output');

tp=cm(1,1);
fp=cm(1,2);
fn=cm(2,1);
tn=cm(2,2);
acc=(tp+tn)/(tp+fp+fn+tn);
spec=(tn)/(fp+tn);
sens=(tp)/(tp+fn);


acc_kfold_svm(i,:)=acc;
spec_kfold_svm(i,:)=spec;
sens_kfold_svm(i,:)=sens;
[~,~,~,AUC] = perfcurve(Laebl_train',output',1);
auc_kfold_svm(i,:)=AUC;


end


K=["k=1";"k=2";"k=3";"k=4";"k=5";"Mean";"std"];
Acc=[acc_kfold_svm*100;mean(acc_kfold_svm);std(acc_kfold_svm)];
Spec=[spec_kfold_svm*100;mean(spec_kfold_svm);std(spec_kfold_svm)];
Sens=[sens_kfold_svm*100;mean(sens_kfold_svm);std(sens_kfold_svm)];
Result_SVM=table(K,Acc,Spec,Sens);

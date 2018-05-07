#some basic configuration

#the number of classes for every datasets
datasets = {
    "UCF":101,
    "HMDB":51,
    "Something":174
}

datasets_path ={
    "UCF":"D://UCF-101",
    "HMDB":"/",
    "SomethingData":"/",
    "SomethingAnnotation":"./something",
    "SomethingLabel":"./something/something-something-v1-labels.csv",
    "SomethingTrain":"./something/something-something-v1-train.csv",
    "SomethingValidation":"./something/something-something-v1-validation.csv",
    "SomethingTest":"./something/something-something-v1-test.csv"

}

pretrained_params_path = {'resnet50':'./resnet50-19c8e357.pth'}

check_points_path = "/"
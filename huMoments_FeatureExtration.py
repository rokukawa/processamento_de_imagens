import os
import cv2
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from progress.bar import Bar
import time

def main():
    mainStartTime = time.time()
    trainImagePath = './images_split/train/'
    testImagePath = './images_split/test/'
    trainFeaturePath = './features_labels/train/'
    testFeaturePath = './features_labels/test/'

    print(f'[INFO] ========= TRAINING IMAGES ========= ')
    trainImages, trainLabels = getData(trainImagePath)
    trainEncodedLabels, encoderClasses = encodeLabels(trainLabels)
    trainFeatures = extractHuMomentsFeatures(trainImages)

    # Treinar e avaliar o classificador
    classifier, accuracy = train_and_evaluate(trainFeatures, trainEncodedLabels)

    print(f'[INFO] Acurácia do classificador no conjunto de treinamento: {accuracy * 100:.2f}%')
    saveData(trainFeaturePath, trainEncodedLabels, trainFeatures, encoderClasses)

    print(f'[INFO] =========== TEST IMAGES =========== ')
    testImages, testLabels = getData(testImagePath)
    testEncodedLabels, encoderClasses = encodeLabels(testLabels)
    testFeatures = extractHuMomentsFeatures(testImages)

    # Avaliar o classificador no conjunto de teste
    test_accuracy = evaluate_classifier(classifier, testFeatures, testEncodedLabels)

    print(f'[INFO] Acurácia do classificador no conjunto de teste: {test_accuracy * 100:.2f}%')

    # Gerar e imprimir a matriz de confusão
    confusion_mat = confusion_matrix(testEncodedLabels, classifier.predict(testFeatures))
    print(f'[INFO] Matriz de Confusão:\n{confusion_mat}')

    saveData(testFeaturePath, testEncodedLabels, testFeatures, encoderClasses)


def train_and_evaluate(features, labels):
    # Dividir os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Inicializar o classificador SVM
    classifier = SVC()

    # Treinar o classificador
    classifier.fit(X_train, y_train)

    # Fazer previsões no conjunto de teste
    predictions = classifier.predict(X_test)

    # Calcular a acurácia
    accuracy = accuracy_score(y_test, predictions)

    return classifier, accuracy

def evaluate_classifier(classifier, features, labels):
    # Fazer previsões no conjunto de teste
    predictions = classifier.predict(features)

    # Calcular a acurácia
    accuracy = accuracy_score(labels, predictions)

    return accuracy

def getData(path):
    images = []
    labels = []
    if os.path.exists(path):
        for dirpath , dirnames , filenames in os.walk(path):   
            if (len(filenames)>0): #it's inside a folder with files
                folder_name = os.path.basename(dirpath)
                bar = Bar(f'[INFO] Getting images and labels from {folder_name}',max=len(filenames),suffix='%(index)d/%(max)d Duration:%(elapsed)ds')            
                for index, file in enumerate(filenames):
                    label = folder_name
                    labels.append(label)
                    full_path = os.path.join(dirpath,file)
                    image = cv2.imread(full_path)
                    images.append(image)
                    bar.next()
                bar.finish()
        #print(labels)
        return images, np.array(labels,dtype=object)
    
def extractHuMomentsFeatures(images):
    bar = Bar('[INFO] Extracting Hu Moments features...', max=len(images), suffix='%(index)d/%(max)d  Duration:%(elapsed)ds')
    featuresList = []

    for image in images:
        if np.ndim(image) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        moments = cv2.moments(image)
        hu_moments = cv2.HuMoments(moments).flatten()
        featuresList.append(hu_moments)
        bar.next()
    bar.finish()
    return np.array(featuresList, dtype=object)

def encodeLabels(labels):
    startTime = time.time()
    print(f'[INFO] Encoding labels to numerical labels')

    encoder = preprocessing.LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    elapsedTime = round(time.time() - startTime,2)

    print(f'[INFO] Encoding done in {elapsedTime}s')
    return np.array(encoded_labels, dtype=int), encoder.classes_

def saveData(path,labels,features,encoderClasses):
    startTime = time.time()
    print(f'[INFO] Saving data')

    label_filename = f'{labels=}'.split('=')[0]+'.csv'
    feature_filename = f'{features=}'.split('=')[0]+'.csv'
    encoder_filename = f'{encoderClasses=}'.split('=')[0]+'.csv'

    np.savetxt(path+label_filename,labels, delimiter=',',fmt='%i')
    np.savetxt(path+feature_filename,features, delimiter=',') #float does not need format
    np.savetxt(path+encoder_filename,encoderClasses, delimiter=',',fmt='%s') 

    elapsedTime = round(time.time() - startTime,2)
    print(f'[INFO] Saving done in {elapsedTime}s')

if __name__ == "__main__":
    main()

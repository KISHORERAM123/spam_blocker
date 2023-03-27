
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

import pickle
from sklearn.feature_extraction.text import CountVectorizer

@csrf_exempt
def predict_spam(request):
    if request.method == 'POST':
        # Get input from the user
        message = request.POST.get('message')
        
        # Load the trained model
        # with open('C:\Users\kisho\tnpolice\tnpolice\hello\spamclassifier.py', 'rb') as f:
        #     clf = pickle.load(f)
        with open(r'C:\Users\kisho\tnpolice\tnpolice\hello\spam_classifier.pkl', 'rb') as f:
            clf = pickle.load(f)
        # print(clf) 
        # Load the vectorizer
        # with open('C:\Users\kisho\tnpolice\tnpolice\hello\vectorizer.pkl', 'rb') as f:
        #     vectorizer = pickle.load(f)
        with open(r'C:\Users\kisho\tnpolice\tnpolice\hello\vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        # print(vectorizer)
        
        # Vectorize the input message
        message_vector = vectorizer.transform([message])
        
        # Make a prediction
        prediction = clf.predict(message_vector)[0]
        
        # Return the prediction as a JSON response
        return JsonResponse({'prediction': prediction})
        
    else:
        return render(request, 'predict_spam.html')

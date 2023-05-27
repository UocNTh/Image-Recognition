import os 
import uuid 
import urllib
from PIL import Image
from flask import Flask, render_template, request , jsonify, send_file
import base64
import tensorflow as tf 
import cv2 
import numpy as np 
from keras.models import load_model
from keras.utils import load_img , img_to_array



app = Flask(__name__ ) 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model1 = tf.keras.models.load_model('mnist.h5')
model2 = load_model(os.path.join(BASE_DIR , 'cifar10.h5')) 

model1.make_predict_function() 

ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png' , 'jfif'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT

classes = ['airplane' ,'automobile', 'bird' , 'cat' , 'deer' ,'dog' ,'frog', 'horse' ,'ship' ,'truck']

def predict(filename , model):
    img = load_img(filename , target_size = (32 , 32))
    img = img_to_array(img)
    img = img.reshape(1 , 32 ,32 ,3)

    img = img.astype('float32')
    img = img/255.0
    result = model.predict(img)

    dict_result = {}
    for i in range(10):
        dict_result[result[0][i]] = classes[i]

    res = result[0]
    res.sort()
    res = res[::-1]
    prob = res[:3]
    
    prob_result = []
    class_result = []
    for i in range(3):
        prob_result.append((prob[i]*100).round(2))
        class_result.append(dict_result[prob[i]])

    return class_result , prob_result


@app.route('/') 

@app.route('/home') 
def home() : 
    return render_template('home.html') 

@app.route('/image_recognition', methods=['POST'])
def image_recognition():
    if 'button1' in request.form:

        return render_template('mnist.html')
    elif 'button2' in request.form:

        return render_template('index.html')
    

@app.route('/recognize', methods = ['POST'])  
def _recognize(): 

    if request.method == 'POST': 
        data = request.get_json()
        imageBase64 = data['image']
        imgBytes = base64.b64decode(imageBase64)

        with open("temp.jpg", "wb" ) as temp : 
            temp.write(imgBytes)

        image = cv2.imread('temp.jpg') 
        image = cv2.resize(image,(28,28), interpolation=cv2.INTER_AREA)
        image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        image_prediction=np.reshape(image_gray, (28,28,1))
        image_prediction=(255-image_prediction.astype('float')) / 255 

        prediction = np.argmax(model1.predict(np.array([image_prediction])), axis=-1) 



        return jsonify({'prediction': str(prediction[0]),
                        'status': True 
                        })
    

@app.route('/success' , methods = ['GET' , 'POST'])

def success():
    error = ''
    target_img = os.path.join(os.getcwd() , 'static/images')
    if request.method == 'POST':
        if(request.form):
            link = request.form.get('link')
            try :
                resource = urllib.request.urlopen(link)
                unique_filename = str(uuid.uuid4())
                filename = unique_filename+".jpg"
                img_path = os.path.join(target_img , filename)
                output = open(img_path , "wb")
                output.write(resource.read())
                output.close()
                img = filename

                class_result , prob_result = predict(img_path , model2)

                predictions = {
                      "class1":class_result[0],
                        "class2":class_result[1],
                        "class3":class_result[2],
                        "prob1": prob_result[0],
                        "prob2": prob_result[1],
                        "prob3": prob_result[2],
                }

            except Exception as e : 
                print(str(e))
                error = 'This image from this site is not accesible or inappropriate input'

            if(len(error) == 0):
                return  render_template('success.html' , img  = img , predictions = predictions)
            else:
                return render_template('index.html' , error = error) 

            
        elif (request.files):
            file = request.files['file']
            if file and allowed_file(file.filename):
                file.save(os.path.join(target_img , file.filename))
                img_path = os.path.join(target_img , file.filename)
                img = file.filename

                class_result , prob_result = predict(img_path , model2)

                predictions = {
                      "class1":class_result[0],
                        "class2":class_result[1],
                        "class3":class_result[2],
                        "prob1": prob_result[0],
                        "prob2": prob_result[1],
                        "prob3": prob_result[2],
                }

            else:
                error = "Please upload images of jpg , jpeg and png extension only"

            if(len(error) == 0):
                return  render_template('success.html' , img  = img , predictions = predictions)
            else:
                return render_template('index.html' , error = error)

    else:
        return render_template('index.html')
    

if __name__ == '__main__' : 
    app.run(debug=True) 
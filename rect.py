### Importing required Libraries
import cv2
import os
from flask import Flask,request,render_template
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.svm import SVC
import pandas as pd
import joblib


### Defining Flask App
app = Flask(__name__)


### Saving Date today in 2 different formats
def datetoday():
    return date.today().strftime("%d-%B-%Y")



### Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX
cap = cv2.VideoCapture(1)


### If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')

if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')

if f'Attendance.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance.csv','w') as f:
        f.write('Name,Roll,Time,Date')


### get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))


### extract the face from an image
def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_points = face_detector.detectMultiScale(gray, 1.3,5)
    return face_points


### Identify face using ML model
def identify_face(facearray):
    return model.predict(facearray)


### A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')

    for user in userlist:
        for imgname in os.listdir(f'static/faces/{ user }' ):
            img = cv2.imread(f'static/faces/{ user }/{ imgname }' )

            resized_face = cv2.resize(img, (50, 50) )
            faces.append(resized_face.ravel() )
            labels.append(user )

    faces = np.array(faces)
    classifier = SVC(kernel='linear')
    classifier.fit(faces,labels)

    joblib.dump(classifier,'static/face_recognition_model.pkl')


### Extract info from today's attendance file in attendance folder
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance.csv')

    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    dates  = df['Date']

    l = len(df)
    return names,rolls,times,dates,l


### Add Attendance of a specific user
def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    current_date = date.today()
    
    df = pd.read_csv(f'Attendance/Attendance.csv')

    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance.csv','a') as f:
            f.write(f'\n{ username },{ userid },{ current_time },{ current_date }')



### Our main page
@app.route('/')
def home():
    names,rolls,times,dates,l = extract_attendance()    
    return render_template('register.html',names=names,rolls=rolls,times=times,dates=dates,l=l,totalreg=totalreg(),datetoday2=datetoday()) 


### This function will run when we click on Attendance Button
@app.route('/start',methods=['GET'])
def start():

    if 'face_recognition_model.pkl' not in os.listdir('static/'):
        return render_template('register.html',totalreg=totalreg(), datetoday2=datetoday(), mess='There is no trained model in the trainer folder. Please add a new face to continue.')

    cap = cv2.VideoCapture(0)
    ret = True

    while ret:
        ret,frame = cap.read()
        frame = cv2.flip(frame,1)
        if extract_faces(frame)!=():
            (x, y, w, h) = extract_faces(frame)[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)

            face = cv2.resize(frame[y:y+h,x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1,-1))[0]
            
            if identified_person =="unknown":
                pass
            else:
                add_attendance(identified_person)
            cv2.putText(frame, f'{identified_person}', (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)

            cv2.imshow('Attendance',frame)
            key=cv2.waitKey(1)
            if key==ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    names,rolls,times,dates,l = extract_attendance()    
    return render_template('register.html', names=names, rolls=rolls, times=times, dates = dates, l=l, totalreg=totalreg(), datetoday=datetoday()) 


### This function will display form to  add a new user
@app.route('/form',methods=['GET','POST'])
def form():
    return render_template('index.html',totalreg=totalreg(), datetoday=datetoday())

### This function will add a new user
@app.route('/add',methods=['GET','POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/' + newusername + '_' + str(newuserid)

    if not os.path.isdir(userimagefolder ):
        os.makedirs(userimagefolder)
    cap = cv2.VideoCapture(0)
    i,j = 0,0

    while True:
        _,frame = cap.read()

        cv2.flip(frame,1)
        faces = extract_faces(frame)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/80', (30,30), font, 1, (255, 0, 20), 2, cv2.LINE_AA)
            if j%10 ==0:
                name = newusername + '_' + str(i) + '.jpg'
                cv2.imwrite(userimagefolder + '/' + name, frame[y:y+h,x:x+w] )
                i+= 1
            j+= 1
        if j==800:
            break
        cv2.imshow('... Adding new User ...', frame)
        key=cv2.waitKey(1)
        if key==ord('q'):
            break

    
    cap.release()
    cv2.destroyAllWindows()

    print('.... Training Model ....')
    if(totalreg() >1):
        train_model()
    names,rolls,times,dates,l = extract_attendance()    
    return render_template('register.html', names=names, rolls=rolls, times=times, dates = dates, l=l, totalreg=totalreg(), datetoday=datetoday() ) 

### To get the previous attendence data
@app.route('/get_attendence',methods=['POST','GET'])
def get_attendence():
    names,rolls,times,dates,l = extract_attendance()
    return render_template('attendence.html',names=names, rolls=rolls, times=times, dates = dates, l=l, totalreg=totalreg(), datetoday=datetoday())

### Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True)
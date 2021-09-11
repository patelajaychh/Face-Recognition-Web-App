import numpy as np
import cv2
from keras.layers import Lambda, Input
from keras.models import Model
from keras import backend as K
import tensorflow as tf
from utils_fr.umeyama import umeyama
from models.verifier.rface.lresnet100e_ir import LResNet100E_IR
import os
import json

def resize_tensor(size):
    input_tensor = Input((None, None, 3)) 
    output_tensor = Lambda(lambda x: tf.compat.v1.image.resize_bilinear(x, [size, size]))(input_tensor)
    return Model(input_tensor, output_tensor)

def l2_norm(latent_dim):            
    input_tensor = Input((latent_dim,))
    output_tensor = Lambda(lambda x: K.l2_normalize(x))(input_tensor)
    return Model(input_tensor, output_tensor)


class SupportMethods(object):
    """
    provides the various methods required for specific tasks.
    """

    def __init__(self):
        """
        provides the various methods required for specific tasks.
        """

    #Added by Arunendra Kumar to insert attendance records in DB
    def insertOrUpdateAttendance(username,photopath):
        status = False
        try:
            connection = psycopg2.connect(user = "postgres",password = "Welcome123",host = "localhost",port = "5432",database = "cdac_employees_attendance")
            cursor = connection.cursor()
            # print ( connection.get_dsn_parameters(),"\n")
            sql = "select count(*) from attendances where employee_id={} and date=now()::date".format(username)
            cursor.execute(sql)
            count= cursor.fetchone()[0];
            print("DB count=>", count)
            status=False
            if count==0:
                sql = "insert into attendances(employee_id,date,stime,sphoto) values('{}',now()::date,now()::time(0),'{photopath}')".format(username)
                cursor.execute(sql)
                connection.commit()
                res= cursor.rowcount
                if res>0:
                    status=True
            else:
                sql = "update attendances set etime= now()::time(0), ephoto='{}' where employee_id='{}' and date=now()::date".format(photopath,username)
                cursor.execute(sql)
                connection.commit()
                res = cursor.rowcount
                if res > 0:
                    status = True
        except (Exception, psycopg2.Error) as error :
            print ("Error while connecting to PostgreSQL", error)

        try:
            if(connection):
                cursor.close()
                connection.close()
        except:
            pass
        return status        

    def putTextWithBG(img, text, cordinates, fontFace, font_scale, font_thickness, text_color=(255,0,0), bg_color = (255,255,255)):

        text_size, _ = cv2.getTextSize(text, fontFace, font_scale, font_thickness)
        text_w, text_h = text_size
        x,y = cordinates
        cv2.rectangle(img, (x,y-10), (x+text_w, y+text_h), bg_color, -1)
        cv2.putText(img, text, (x,y), fontFace=fontFace, fontScale=font_scale, color=text_color, thickness=font_thickness)
        
        return text_size

    def read_json_from_paths(enrollfolder_JSON_path, embed_dimension=512):
      '''
      Reads json files and return numpy array of labels and embeds
      embed_paths: json files paths
      embed_dimension: Feature vetor dimension
      return: embeds and labels
      '''
      embeds = []
      labels = []
      names = []
      #try:
      folders = os.listdir(enrollfolder_JSON_path)
      for fold in os.listdir(enrollfolder_JSON_path):
        jsons = os.listdir(os.path.join(enrollfolder_JSON_path, fold, "JSON"))
        for jsn in jsons:
          with open(os.path.join(enrollfolder_JSON_path, fold, "JSON", jsn)) as f:
              emb = json.load(f)['data']
              embeds.append(emb)
              labels.append(fold)
              names.append(jsn)
  
      #except:
      #  pass
      embeds = np.array(embeds, dtype='float32')
      embeds = embeds.reshape(embeds.shape[0], embed_dimension)
      
      return embeds, np.asarray(labels), names
    
    
    def resize_image(im, max_size=768):
      if np.max(im.shape) > max_size:
          ratio = max_size / np.max(im.shape)
          print("Resize image to ({str(int(im.shape[1]*ratio))}, {str(int(im.shape[0]*ratio))}).")
          return cv2.resize(im, (0,0), fx=ratio, fy=ratio)
      return im

    def align_face(im, src, size):
        dst = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041] ], dtype=np.float32 )
        dst[:,0] += 8.0       
        dst = dst / 112 * size 
        M = umeyama(src, dst, True)[0:2]
        warped = cv2.warpAffine(im, M, (size, size), borderValue=0.0)
        return warped 

    def create_mtcnn():
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
            #gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5)
            #sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))  
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)) #tf.compat.v1.ConfigProto
            with sess.as_default():
                print('Creating networks and loading parameters')
                pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
    
                return pnet, rnet, onet        
            
    def build_rface():
        classes = 512
        latent_dim = classes
        input_resolution = 112
        weights_path = "./models/verifier/rface/lresnet100e_ir_keras.h5"            
        lresnet100e_ir = LResNet100E_IR(weights_path=weights_path)
    
        input_tensor = Input((None, None, 3))
        resize_layer = resize_tensor(size=input_resolution)
        l2_normalize = l2_norm(latent_dim)
                
        output_tensor = l2_normalize(lresnet100e_ir(resize_layer(input_tensor)))  
        net = Model(input_tensor, output_tensor)
        net.trainable = False
        print("Network Built...")
        return net

    def prewhiten(x):
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1/std_adj)
        return y
        
        
    def get_embedding_distance(emb1, emb2):
        #dist = np.sqrt(np.sum(np.square(np.subtract(emb1, emb2))))
        dist = np.linalg.norm(emb1-emb2)
        return dist


    def record_search(test_embeds, features, k=1):
        #print('Calculating Distances.....')
        nrof_test = len(test_embeds)
        nrof_ident = len(features)
    
        D = np.zeros((nrof_test, k))
        I = np.ones((nrof_test, k), dtype='uint32')
        for i in range(len(test_embeds)):
            test_emb = test_embeds[i]
            dist = {}
            for p in range(len(features)):
                
                dist[p] = np.linalg.norm(test_emb - features[p])
                
            sorted_dict = {key:value for key,value in sorted(dist.items(), key= lambda kv: (kv[1], kv[0]))}    
            I[i,:] = list(sorted_dict.keys())[:k]
            D[i,:] = list(sorted_dict.values())[:k]
        
        return D,I

    def record_search1(test_emb, features):
        #print('Calculating Distances.....')
    
        dist = []
        for p in range(len(features)):
          #print(len(test_emb), len(features[p]))
          dist.append(np.linalg.norm(test_emb - features[p]))
                
        I  = dist.index(min(dist)) 
        D  = dist[I]
        #print(dist)
        return D,I    
    
    def detect_rface(detector, img, thresh, do_flip):
        count = 1
        im_shape = img.shape
        scales = [640, 1080]
        target_size = scales[0]
        max_size = scales[1]
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        #im_scale = 1.0
        #if im_size_min>target_size or im_size_max>max_size:
        im_scale = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
        
        #print('im_scale', im_scale)
        
        scales = [im_scale]
        for c in range(count):
          bboxes, landmarks = detector.detect(img, thresh, scales=scales, do_flip=False)
        
          
        return bboxes, landmarks

    def get_largest_face_rface(bboxes,landmarks ):
    
        #num_faces = len(bboxes)
        bounding_box_size = (bboxes[:,2]-bboxes[:,0])*(bboxes[:,3]-bboxes[:,1])  
        indx = np.argmax(bounding_box_size)
        bboxes = bboxes[indx]
        
        points = landmarks[indx]
        landmarks = np.zeros((5,2), dtype=float)
        landmarks[:,0] = points[:,1]
        landmarks[:,1] = points[:,0]
        landmarks = np.asarray([landmarks])
    
        return bboxes, landmarks
        
    
    def detect_face_mtcnn(img, pnet, rnet, onet):
    
        pnet, rnet, onet = SupportMethods.create_mtcnn()
        detect_multiple_faces = 0
        minsize = 50 # minimum size of face
        threshold =  [0.6, 0.7, 0.7]  # three steps's threshold #    [ 0.6, 0.7, 0.9 ]
        factor = 0.85 # scale factor
        bboxes, landmarks = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        points =  landmarks.reshape( (2,5) ).T
        landmarks = np.zeros((5,2), dtype=float)
        landmarks[:,0] = points[:,1]
        landmarks[:,1] = points[:,0]
        landmarks = np.asarray([landmarks])
        
        return bboxes, landmarks
    
    def prepare_landmarks_with_margin(landmarks, margin):
    
        ## Left Eye
        landmarks[0][0][0] = np.maximum(landmarks[0][0][0] - margin/2,0)  # Y Component
        landmarks[0][0][1] = np.maximum(landmarks[0][0][1] - margin/2,0)  # X Component
        ## Right Eye        
        landmarks[0][1][0] = np.maximum(landmarks[0][1][0] - margin/2, 0)  # Y Component
        landmarks[0][1][1] = np.minimum(landmarks[0][1][1] + margin/2, imagedata.shape[1])  # X Component
        ## Left Mouth
        landmarks[0][3][0] = np.minimum(landmarks[0][3][0] + margin/2, imagedata.shape[1])  # Y Component
        landmarks[0][3][1] = np.maximum(landmarks[0][3][1] - margin/2, 0)  # X Component
        ## right Mouth
        landmarks[0][4][0] = np.minimum(landmarks[0][4][0] + margin/2, imagedata.shape[1])  # Y Component
        landmarks[0][4][1] = np.minimum(landmarks[0][4][1] + margin/2, imagedata.shape[1])  # X Component
    
        return landmarks
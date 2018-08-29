#!/home/rafal/anaconda3/envs/pyannote/bin/python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2015-2017 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr

"""Face detection and tracking

The standard pipeline is the following

      face tracking => feature extraction => face clustering

General options:

  --ffmpeg=<ffmpeg>         Specify which `ffmpeg` to use.
  -h --help                 Show this screen.
  --version                 Show version.
  --verbose                 Show processing progress.

Face tracking options (track):

  <video>                   Path to video file.
  <shot.json>               Path to shot segmentation result file.
  <tracking>                Path to tracking result file.

  --min-size=<ratio>        Approximate size (in video height ratio) of the
                            smallest face that should be detected. Default is
                            to try and detect any object [default: 0.0].
  --every=<seconds>         Only apply detection every <seconds> seconds.
                            Default is to process every frame [default: 0.0].
  --min-overlap=<ratio>     Associates face with tracker if overlap is greater
                            than <ratio> [default: 0.5].
  --min-confidence=<float>  Reset trackers with confidence lower than <float>
                            [default: 10.].
  --max-gap=<float>         Bridge gaps with duration shorter than <float>
                            [default: 1.].

Feature extraction options (features):

  <video>                   Path to video file.
  <tracking>                Path to tracking result file.
  <landmark_model>          Path to dlib facial landmark detection model.
  <embedding_model>         Path to dlib feature extraction model.
  <landmarks>               Path to facial landmarks detection result file.
  <embeddings>              Path to feature extraction result file.

Visualization options (demo):

  <video>                   Path to video file.
  <tracking>                Path to tracking result file.
  <output>                  Path to demo video file.

  --height=<pixels>         Height of demo video file [default: 400].
  --from=<sec>              Encode demo from <sec> seconds [default: 0].
  --until=<sec>             Encode demo until <sec> seconds.
  --shift=<sec>             Shift result files by <sec> seconds [default: 0].
  --landmark=<path>         Path to facial landmarks detection result file.
  --label=<path>            Path to track identification result file.

"""

from __future__ import division

from pyannote.core import Annotation
import pyannote.core.json
from pyannote.core import Timeline

from pyannote.video import __version__
from pyannote.video import Video
from pyannote.video import Face
from pyannote.video import FaceTracking
from pyannote.video import Shot, Thread
from pyannote.video.face.clustering import FaceClustering

from pandas import read_table
import pandas as pd

from six.moves import zip
import numpy as np
import cv2

import dlib
import tensorflow as tf
from facenet.src import facenet
from facenet.src.align.detect_face import bulk_detect_face, create_mtcnn
from defaults import argHandler
import logging



FACE_TEMPLATE = ('{t:.3f} {identifier:d} '
                 '{left:.3f} {top:.3f} {right:.3f} {bottom:.3f} '
                 '{status:s}\n')




def do_shot(video, output, height=50, window=2.0, threshold=1.0):

    shots = Shot(video, height=height, context=window, threshold=threshold)
    shots = Timeline(shots)
    with open(output, 'w') as fp:
        pyannote.core.json.dump(shots, fp)

def getFaceGenerator(tracking, frame_width, frame_height, double=True):
    """Parse precomputed face file and generate timestamped faces"""

    # load tracking file and sort it by timestamp
    names = ['t', 'track', 'left', 'top', 'right', 'bottom', 'status']
    dtype = {'left': np.float32, 'top': np.float32,
             'right': np.float32, 'bottom': np.float32}
    tracking = read_table(tracking, delim_whitespace=True, header=None,
                          names=names, dtype=dtype)
    tracking = tracking.sort_values('t')

    # t is the time sent by the frame generator
    t = yield

    rectangle = dlib.drectangle if double else dlib.rectangle

    faces = []
    currentT = None

    for _, (T, identifier, left, top, right, bottom, status) in tracking.iterrows():

        left = int(left * frame_width)
        right = int(right * frame_width)
        top = int(top * frame_height)
        bottom = int(bottom * frame_height)

        face = rectangle(left, top, right, bottom)

        # load all faces from current frame and only those faces
        if T == currentT or currentT is None:
            faces.append((identifier, face, status))
            currentT = T
            continue

        # once all faces at current time are loaded
        # wait until t reaches current time
        # then returns all faces at once

        while True:

            # wait...
            if currentT > t:
                t = yield t, []
                continue

            # return all faces at once
            t = yield currentT, faces

            # reset current time and corresponding faces
            faces = [(identifier, face, status)]
            currentT = T
            break

    while True:
        t = yield t, []


def pairwise(iterable):
    "s -> (s0,s1), (s2,s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)


def getLandmarkGenerator(shape, frame_width, frame_height):
    """Parse precomputed shape file and generate timestamped shapes"""

    # load landmarks file
    shape = read_table(shape, delim_whitespace=True, header=None)

    # deduce number of landmarks from file dimension
    _, d = shape.shape
    n_points = (d - 2) / 2

    # t is the time sent by the frame generator
    t = yield

    shapes = []
    currentT = None

    for _, row in shape.iterrows():

        T = float(row[0])
        identifier = int(row[1])
        landmarks = np.float32(list(pairwise(
            [coordinate for coordinate in row[2:]])))
        landmarks[:, 0] = np.round(landmarks[:, 0] * frame_width)
        landmarks[:, 1] = np.round(landmarks[:, 1] * frame_height)

        # load all shapes from current frame
        # and only those shapes
        if T == currentT or currentT is None:
            shapes.append((identifier, landmarks))
            currentT = T
            continue

        # once all shapes at current time are loaded
        # wait until t reaches current time
        # then returns all shapes at once

        while True:

            # wait...
            if currentT > t:
                t = yield t, []
                continue

            # return all shapes at once
            t = yield currentT, shapes

            # reset current time and corresponding shapes
            shapes = [(identifier, landmarks)]
            currentT = T
            break

    while True:
        t = yield t, []


def track(video, output,
          detect_min_size=0.0,
          detect_every=0.0,
          track_min_overlap_ratio=0.5,
          track_min_confidence=10.,
          track_max_gap=1.):
    """Tracking by detection"""

    tracking = FaceTracking(detect_min_size=detect_min_size,
                            detect_every=detect_every,
                            track_min_overlap_ratio=track_min_overlap_ratio,
                            track_min_confidence=track_min_confidence,
                            track_max_gap=track_max_gap)

    track_l = []

    with open(output, 'w') as foutput:

        for identifier, track in enumerate(tracking(video)):

            for t, (left, top, right, bottom), status in track:

                foutput.write(FACE_TEMPLATE.format(
                    t=t, identifier=identifier, status=status,
                    left=left, right=right, top=top, bottom=bottom))

                track_l.append({"t":t, "identifier":identifier, "status":status,
                    "left":left, "right":right, "top":top, "bottom":bottom})

            foutput.flush()
            # track_pd = pd.Dataframe(track_l, columns=["t", "identifier":identifier, "status":status,
            #         "left":left, "right":right, "top":top, "bottom":bottom])


def track2(video, output,
          detect_min_size=0.0,
          detect_every=0.0,
          track_min_overlap_ratio=0.5,
          track_min_confidence=10.,
          track_max_gap=1.):
    """Tracking by detection"""

    # tracking = FaceTracking(detect_min_size=detect_min_size,
    #                         detect_every=detect_every,
    #                         track_min_overlap_ratio=track_min_overlap_ratio,
    #                         track_min_confidence=track_min_confidence,
    #                         track_max_gap=track_max_gap)

    # track_l = []


        # init tracker
    tracker = Sort()  # create instance of the SORT tracker

    with tf.Graph().as_default():
        with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True), log_device_placement=False)) as sess:
            pnet, rnet, onet = detect_face.create_mtcnn(sess, os.path.join(project_dir, "align"))

            margin = 10  # if the face is big in your video ,you can set it bigger for tracking easiler
            minsize = 100  # minimum size of face for mtcnn to detect
            threshold = [0.6, 0.7, 0.7]  # three steps's threshold
            factor = 0.709  # scale factor
            frame_interval = 5  # interval how many frames to make a detection,you need to keep a balance between performance and fluency
            scale_rate = 1.  # if set it smaller will make input frames smaller



            for i, (t, frame) in enumerate(video):
                final_faces = []
                addtional_attribute_list = []
                frame = cv2.resize(frame, (0, 0), fx=scale_rate, fy=scale_rate)
                r_g_b_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if c % frame_interval == 0:
                    img_size = np.asarray(frame.shape)[0:2]
                    faces, points = detect_face.detect_face(r_g_b_frame, minsize, pnet, rnet, onet, threshold, factor)
                    face_sums = faces.shape[0]
                    if face_sums > 0:
                        face_list = []
                        for i, item in enumerate(faces):
                            f = round(faces[i, 4], 6)
                            if f > 0.99:
                                det = np.squeeze(faces[i, 0:4])

                                # face rectangle
                                det[0] = np.maximum(det[0] - margin, 0)
                                det[1] = np.maximum(det[1] - margin, 0)
                                det[2] = np.minimum(det[2] + margin, img_size[1])
                                det[3] = np.minimum(det[3] + margin, img_size[0])
                                face_list.append(item)

                                # face cropped
                                bb = np.array(det, dtype=np.int32)
                                frame_copy = frame.copy()
                                cropped = frame_copy[bb[1]:bb[3], bb[0]:bb[2], :]

                                # use 5 face landmarks  to judge the face is front or side
                                squeeze_points = np.squeeze(points[:, i])
                                tolist = squeeze_points.tolist()
                                facial_landmarks = []
                                for j in range(5):
                                    item = [tolist[j], tolist[(j + 5)]]
                                    facial_landmarks.append(item)
                                if args.face_landmarks:
                                    for (x, y) in facial_landmarks:
                                        cv2.circle(frame_copy, (int(x), int(y)), 3, (0, 255, 0), -1)
                                dist_rate, high_ratio_variance, width_rate = judge_side_face(
                                    np.array(facial_landmarks))

                                # face addtional attribute(index 0:face score; index 1:0 represents front face and 1 for side face )
                                item_list = [cropped, faces[i, 4], dist_rate, high_ratio_variance, width_rate]
                                addtional_attribute_list.append(item_list)

                        final_faces = np.array(face_list)

                trackers = tracker.update(final_faces, img_size, directoryname, addtional_attribute_list, r_g_b_frame)

                c += 1

                for d in trackers:
                    if display:
                        d = d.astype(np.int32)
                        cv2.rectangle(frame, (d[0], d[1]), (d[2], d[3]), colours[d[4] % 32, :] * 255, 5)
                        cv2.putText(frame, 'ID : %d' % (d[4]), (d[0] - 10, d[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.75,
                                    colours[d[4] % 32, :] * 255, 2)
                        if final_faces != []:
                            cv2.putText(frame, 'DETECTOR', (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                        (1, 1, 1), 2)

    with open(output, 'w') as foutput:

        for identifier, track in enumerate(tracking(video)):

            for t, (left, top, right, bottom), status in track:

                foutput.write(FACE_TEMPLATE.format(
                    t=t, identifier=identifier, status=status,
                    left=left, right=right, top=top, bottom=bottom))

                track_l.append({"t":t, "identifier":identifier, "status":status,
                    "left":left, "right":right, "top":top, "bottom":bottom})

            foutput.flush()
            # track_pd = pd.Dataframe(track_l, columns=["t", "identifier":identifier, "status":status,
            #         "left":left, "right":right, "top":top, "bottom":bottom])


def get_face(frame, box, size, margin=0):
    frame_height, frame_width = frame.shape[:2]
    top, left, bottom, right = box.top(), box.left(), box.bottom(), box.right()
    max_dim = max(bottom-top, right-left)
    max_dim = min(max_dim, frame_width, frame_height)
    if right>frame_width or bottom>frame_height or top>=bottom or left>=right:
        return None
    y_add = (max_dim-(bottom-top))
    x_add = (max_dim-(right-left))
    if y_add!=0:
        top = top - y_add//2
        bottom = bottom + max_dim - (bottom - top)
    if x_add!=0:
        left = left - x_add//2
        right = right + max_dim - (right - left)
    if left<0:
        right = right - left
        left = 0
    if top<0:
        bottom = bottom-top
        top=0
    if right>frame_width:
        left=left - (right-frame_width)
        right = frame_width
    if bottom>frame_height:
        top = top - (bottom-frame_height)
        bottom = frame_height

    img_base = frame[top:bottom, left:right]
    

    return cv2.resize(img_base, (size, size))



    
def extract_old(video, landmark_model, embedding_model, tracking, landmark_output, embedding_output):
    """Facial features detection"""

    # face generator
    frame_width, frame_height = video.frame_size
    faceGenerator = getFaceGenerator(tracking,
                                     frame_width, frame_height,
                                     double=False)
    faceGenerator.send(None)

    face = Face(landmarks=landmark_model,
                embedding=embedding_model)

    with open(landmark_output, 'w') as flandmark, \
         open(embedding_output, 'w') as fembedding:

        for timestamp, rgb in video:

            # get all detected faces at this time
            T, faces = faceGenerator.send(timestamp)
            # not that T might be differ slightly from t
            # due to different steps in frame iteration

            for identifier, bounding_box, _ in faces:

                landmarks = face.get_landmarks(rgb, bounding_box)
                embedding = face.get_embedding(rgb, landmarks)

                flandmark.write('{t:.3f} {identifier:d}'.format(
                    t=T, identifier=identifier))
                for p in landmarks.parts():
                    x, y = p.x, p.y
                    flandmark.write(' {x:.5f} {y:.5f}'.format(x=x / frame_width,
                                                            y=y / frame_height))
                flandmark.write('\n')

                fembedding.write('{t:.3f} {identifier:d}'.format(
                    t=T, identifier=identifier))
                for x in embedding:
                    fembedding.write(' {x:.5f}'.format(x=x))
                fembedding.write('\n')

            flandmark.flush()
            fembedding.flush()

def extract(video, landmark_model, embedding_model, tracking, landmark_output, embedding_output):
    """Facial features detection"""

    def run_model():
        face_arr = np.stack(faces_bulk)
        feed_dict = {images_placeholder: face_arr, phase_train_placeholder: False}
        emb = sess.run(embeddings, feed_dict=feed_dict)

        for i in range(len(identifier_bulk)):
            fembedding.write('{t:.3f} {identifier:d}'.format(
                t=t_bulk[i], identifier=identifier_bulk[i]))
            for x in emb[i]:
                fembedding.write(' {x:.5f}'.format(x=x))
            fembedding.write('\n')
        fembedding.flush()
        faces_bulk.clear()
        identifier_bulk.clear()
        t_bulk.clear()

    # face generator
    frame_width, frame_height = video.frame_size
    faceGenerator = getFaceGenerator(tracking,
                                     frame_width, frame_height,
                                     double=False)
    faceGenerator.send(None)

    face = Face(landmarks=landmark_model,
                embedding=embedding_model)

    faces_bulk = []
    identifier_bulk = []
    t_bulk = []
    extract_batch_size = 1000

    with tf.Graph().as_default():

        with tf.Session() as sess:
            facenet.load_model('/home/rafal/nabu/nabu_datascience/facenet/models/face993')

            images_placeholder = sess.graph.get_tensor_by_name("input:0")
            embeddings = sess.graph.get_tensor_by_name("embeddings:0")
            phase_train_placeholder = sess.graph.get_tensor_by_name("phase_train:0")

            with open(landmark_output, 'w') as flandmark, \
                open(embedding_output, 'w') as fembedding:

                #for timestamp, rgb in video:
                for i, (t, frame) in enumerate(video):

                    # get all detected faces at this time
                    T, faces = faceGenerator.send(t)
                    # not that T might be differ slightly from t
                    # due to different steps in frame iteration

                    for identifier, bounding_box, _ in faces:

                        face_resized = get_face(frame, bounding_box, 160)
                        if face_resized is None:
                            continue

                        #cv2.imwrite('tmp.jpg', face_resized[:,:,::-1])
                        #cv2.imwrite('frame.jpg', frame[:,:,::-1])
                        faces_bulk.append(face_resized[:,:,::-1])
                        identifier_bulk.append(identifier)
                        t_bulk.append(T)
                        if len(identifier_bulk) == extract_batch_size:
                            run_model()
                if len(identifier_bulk)>0:            
                    run_model()


def get_make_frame(video, tracking, landmark=None, labels=None,
                   height=200, shift=0.0):
    '''
    COLORS = [
        (240, 163, 255), (  0, 117, 220), (153,  63,   0), ( 76,   0,  92),
        ( 25,  25,  25), (  0,  92,  49), ( 43, 206,  72), (255, 204, 153),
        (128, 128, 128), (148, 255, 181), (143, 124,   0), (157, 204,   0),
        (194,   0, 136), (  0,  51, 128), (255, 164,   5), (255, 168, 187),
        ( 66, 102,   0), (255,   0,  16), ( 94, 241, 242), (  0, 153, 143),
        (224, 255, 102), (116,  10, 255), (153,   0,   0), (255, 255, 128),
        (255, 255,   0), (255,  80,   5)
    ]

    sex_ages = {'0':'M, 43y',
                '3':'M, 47y',
                '6':'F, 45y',
                '8':'M, 35y',
                '17':'F, 30y',
                '14':'M, 42y',
                '24':'M, 31y',
                '19':'M, 29y',
                '28':'M, 66y',
                '37':'M, 52y',
                '35':'F, 43y',
                '41':'M, 33y',
                '43':'M, 26y',
                '56':'F, 42y',
                '62':'M, 35y',
                '52':'M, 29y',
                '58':'M, 33y',
                '67':'M, 37y',
                '79':'F, 32y',
                '73':'M, 29y',
                '82':'M, 28y',
                '76':'M, 36y',
                '86':'M, 29y',
                '97':'M, 23y',
                '63':'M, 28y'
                }

    names = {'0':'John',
                '3':'Bob',
                '6':'Sue',
                '8':'Oliver',
                '17':'Claire',
                '14':'Harry',
                '24':'Jack',
                '19':'Charlie',
                '28':'Tom',
                '37':'Alex',
                '35':'Jess',
                '41':'Ethan',
                '43':'Fred',
                '56':'Anne',
                '62':'James',
                '52':'Ed',
                '58':'Arthur',
                '67':'Jake',
                '79':'Kate',
                '73':'Ted',
                '82':'Luke',
                '76':'Frank',
                '86':'Ryan',
                '97':'Max',
                '63':'Leon'
                }
    '''
    video_width, video_height = video.size
    ratio = height / video_height
    width = int(ratio * video_width)
    video.frame_size = (width, height)

    faceGenerator = getFaceGenerator(tracking, width, height, double=True)
    faceGenerator.send(None)

    if landmark:
        landmarkGenerator = getLandmarkGenerator(landmark, width, height)
        landmarkGenerator.send(None)

    if labels is None:
        labels = dict()

    def make_frame(t):

        frame = video(t)
        _, faces = faceGenerator.send(t - shift)

        if landmark:
            _, landmarks = landmarkGenerator.send(t - shift)

        cv2.putText(frame, '{t:.3f}'.format(t=t), (10, height-10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (175, 238, 238), 1, 8, False)
        for i, (identifier, face, _) in enumerate(faces):
            #color = COLORS[identifier % len(COLORS)]
            color = (175, 238, 238)



            # Draw face bounding box
            pt1 = (int(face.left()), int(face.top()))
            pt2 = (int(face.right()), int(face.bottom()))
            cv2.rectangle(frame, pt1, pt2, color, 1)

            #Print tracker identifier
            # cv2.putText(frame, '#{identifier:d}'.format(identifier=identifier),
            #             (pt1[0], pt2[1] + 15), cv2.FONT_HERSHEY_SIMPLEX,
            #             0.5, color, 1, 8, False)

            # Print track label
            label = labels.get(identifier, '')
            #add sex and age info
            sex_age = sex_ages.get(label, '')
            #name
            name = names.get(label, '')

            cv2.putText(frame,
                        '{label:s}'.format(label=name),
                        (pt1[0], pt1[1] - 7), cv2.FONT_HERSHEY_SIMPLEX,
                        0.333, color, 1, 8, False)

            cv2.putText(frame, '{sex_age:s}'.format(sex_age=sex_age),
                        (pt1[0], pt2[1] + 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.333, color, 1, 8, False)

            # Draw nose
            if landmark:
                points = landmarks[i][0].parts()
                pt1 = (int(points[27, 0]), int(points[27, 1]))
                pt2 = (int(points[33, 0]), int(points[33, 1]))
                cv2.line(frame, pt1, pt2, color, 1)

        return frame

    return make_frame

def demo(filename, tracking, output, t_start=0., t_end=None, shift=0.,
         labels=None, landmark=None, height=200, ffmpeg=None):

    # parse label file
    if labels is not None:
        with open(labels, 'r') as f:
            labels = {}
            for line in f:
                identifier, label = line.strip().split()
                identifier = int(identifier)
                labels[identifier] = label

    video = Video(filename, ffmpeg=ffmpeg)

    from moviepy.editor import VideoClip, AudioFileClip

    make_frame = get_make_frame(video, tracking, landmark=landmark,
                                labels=labels, height=height, shift=shift)
    clip = VideoClip(make_frame, duration=video.duration)
    #audio_clip = AudioFileClip(filename)
    #clip = video_clip.set_audio(audio_clip)

    if t_end is None:
        t_end = video.duration

    clip = clip.subclip(t_start, t_end)
    clip.write_videofile(output, fps=video.frame_rate)


if __name__ == '__main__':

    # parse command line arguments
    version = 'pyannote-face {version}'.format(version=__version__)
    # arguments = docopt(__doc__, argv=['demo', '../cv_videos/P2E_S5_C2.avi',
    #                    '../cv_videos/P2E_S5_C2.track.txt',
    #                    '../cv_videos/P2E_S5_C2.track.mp4'] ,version=version)

    arguments = argHandler()
    arguments.setDefaults()


    videos = ['sample/P2_S5_C2.avi']


    for filename in videos[:1]:

        # initialize video
        #filename = arguments['<video>']
        ffmpeg = arguments['--ffmpeg']
        verbose = arguments['--verbose']

        video = Video(filename, ffmpeg=ffmpeg, verbose=verbose)

        # face tracking arguments
        shot = arguments['<shot.json>']
        tracking = arguments['<tracking>']
        detect_min_size = float(arguments['--min-size'])
        detect_every = float(arguments['--every'])
        track_min_overlap_ratio = float(arguments['--min-overlap'])
        track_min_confidence = float(arguments['--min-confidence'])
        track_max_gap = float(arguments['--max-gap'])


        # facial features detection
        landmark_model = arguments['<landmark_model>']
        embedding_model = arguments['<embedding_model>']
        landmarks = arguments['<landmarks>']
        embeddings = arguments['<embeddings>']
        

        #visualization
        output = arguments['<output>']
        t_start = float(arguments['--from'])
        t_end = arguments['--until']
        t_end = float(t_end) if t_end else None
        shift = float(arguments['--shift'])
        labels = arguments['--label']
        if not labels:
            labels = None
        landmark = arguments['--landmark']
        if not landmark:
            landmark = None
        height = arguments['--height']
        if height is None:
            height = video._height
        
        track(video, tracking,
                detect_min_size=detect_min_size,
                detect_every=detect_every,
                track_min_overlap_ratio=track_min_overlap_ratio,
                track_min_confidence=track_min_confidence,
                track_max_gap=track_max_gap)

        
        
        demo(filename, tracking, output,
                t_start=t_start, t_end=t_end,
                landmark=landmark, height=height,
                shift=shift, ffmpeg=ffmpeg)
        
        
        extract_old(video, landmark_model, embedding_model, tracking,
            landmarks, embeddings)
        
        
        # for cosine=0.09, euclidean=0.6
        clustering = FaceClustering(threshold=0.6)
        face_tracks, emb = clustering.model.preprocess(embeddings)
        #face_tracks.get_timeline()
        import time
        tic = time.time()
        result = clustering(face_tracks, features=emb)
        toe = time.time()
        print(f'time passed: {toe-tic}')

        # from pyannote.core import notebook, Segment
        # notebook.reset()
        # notebook.crop = Segment(0, 30)
        #result = result.rename_labels(mapping=mapping)
        
        with open(labels, 'w') as fp:
            for _, track_id, cluster in result.itertracks(yield_label=True):
                fp.write(f'{track_id} {cluster}\n')
        

        demo(filename, tracking, output,
                t_start=t_start, t_end=t_end,
                landmark=None, height=height,
                shift=shift, labels=labels, ffmpeg=ffmpeg)







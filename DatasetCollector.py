from ultralytics import YOLO
import cv2
import os


def are_training_faces_already_collected_from_videos(target_path: str,
                                                     identities: list,
                                                     number_of_sampled_frames) -> bool:
    # First we check whether the folder for the first identity even exists...
    check_folder = target_path + "/" + identities[0]
    if not os.path.exists(check_folder):
        return False

    # The folder exists, now we retrieve the number of jpg files in the folder
    files = list(filter(lambda x: x.endswith(".jpg"), os.listdir(check_folder)))
    # If the number of jpg files in the training folder is equal to the desired number of sampled frames, then data
    # collection has already taken place.
    if len(files) == number_of_sampled_frames:
        return True

    return False


def get_number_of_frames_in_video(video: str) -> int:
    cap = cv2.VideoCapture(video)
    # Check if video is successfully opened...
    if not cap.isOpened():
        # Oops...there is a problem with opening the file
        raise IOError("Error opening the video!")

    # Now we go through the video and extract up to 500 face snapshots and write these to the
    # training library
    count = 0
    at_end = False
    while cap.isOpened() & at_end == False:
        # We read a frame from video until the end of the file is hit or we have enough snapshots
        ret, img = cap.read()
        if not ret:
            at_end = True
        else:
            count = count + 1

    # Close the file
    cap.release()

    return count


class DatasetCollector:

    def __init__(self):
        # Load a pretrained YOLO model for face detection
        self.__face_detection_model = YOLO("best.pt", verbose=False)
        os.environ["YOLO_VERBOSE"] = "False"

    def collect_faces_from_videos(self, target_path: str, videos: list, identities: list, number_of_frame_samples):
        # We loop through the list of training videos
        for i in range(0, len(identities)):
            # First, we determine the number of frames in the video
            frame_count = get_number_of_frames_in_video(videos[i])
            # We then, determine the frame rate...this is the sampling rate for the training video
            # We want to get frames from across the whole video.
            frame_rate = round(frame_count / number_of_frame_samples / 5, 0)
            # If frame rate is below 1, then we have insufficient frames...
            if frame_rate < 1:
                raise IOError("Insufficient number of frames in video")

            print("Frame rate for video", videos[i], "=", frame_rate)
            # We make a folder for the identity...
            target_identity_path = target_path + "/" + identities[i]
            os.mkdir(target_identity_path)

            # Next, we collect the right number of samples from across the video
            self.collect_faces_from_video(i, videos[i], frame_rate, number_of_frame_samples, target_identity_path)

    def collect_faces_from_video(self, sequence, video, frame_rate, number_of_frame_samples, target_identity_path):
        cap = cv2.VideoCapture(video)
        # Check if video is successfully opened...
        if not cap.isOpened():
            # Oops...there is a problem with opening the file
            raise IOError("Error opening the video!")
        # Each video corresponds to one identity and we link a unique integer to that identity
        face_id = str(sequence)
        # Now we go through the video and extract the face snapshots from across the video and write these to the
        # training path
        frame_counter = 0
        sequence_number = 0
        at_end = False
        while cap.isOpened() & at_end == False:
            # We read a frame from video until the end of the file is hit or we have enough snapshots
            ret, img = cap.read()
            if (not ret) or (sequence_number == number_of_frame_samples):
                at_end = True
            else:
                face_shot = self.get_bounding_boxed_face_shot(img)
                if face_shot is not None:
                    # We have an extracted face...is it a frame that we want to save?
                    if frame_counter % frame_rate == 0:
                        print("Frame counter =", frame_counter, "Sequence number =", sequence_number, "frame_rate =", frame_rate)
                        cv2.imwrite(
                            target_identity_path + "/Identity." + str(face_id) + '.' + str(sequence_number) + ".jpg",
                            face_shot)
                        sequence_number += 1

                    frame_counter += 1

        # Close the file
        cap.release()

    def get_bounding_boxed_face_shot(self, img):
        # The image is turned into a gray scale image for easier face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        # Detect the faces in the image...
        results = self.__face_detection_model(img)
        faces = results[0].boxes
        rectangle = None
        if faces:
            # A face is detected, so we take the first face...
            box = faces[0]
            top_left_x = int(box.xyxy.tolist()[0][0])
            top_left_y = int(box.xyxy.tolist()[0][1])
            bottom_right_x = int(box.xyxy.tolist()[0][2])
            bottom_right_y = int(box.xyxy.tolist()[0][3])
            # We extract the face...
            rectangle = img[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
            # And standardize the size of each extracted face...
            rectangle = cv2.resize(rectangle, (273, 368), cv2.INTER_AREA)

        return rectangle

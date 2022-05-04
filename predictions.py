import os
import tqdm
import cv2


def make_predictions(face_detector, input_dir, output_dir):
    events = os.listdir(input_dir)

    pbar = tqdm.tqdm(events)
    for event in pbar:
        pbar.set_description('Processing event', event)

        event_pred_dir = os.path.join(output_dir, event)
        if not os.path.exists(event_pred_dir):
            os.mkdir(event_pred_dir)

        os.chdir(event_pred_dir)

        event_dir = os.path.join(input_dir, event)
        event_images = os.listdir(event_dir)
        for img_name in event_images:
            img_path = os.path.join(event_dir, img_name)
            image_data = cv2.imread(img_path)
            predictions = face_detector.detect_face(image_data)

            img_name_txt = img_name.rstrip('.jpg') + '.txt'
            with open(img_name_txt, 'w+') as f:
                f.write(img_name.rstrip('.jpg') + '\n')
                f.write(str(len(predictions)) + '\n')
                for pred in predictions:
                    f.write(f"{pred[0]} {pred[1]} {pred[2]} {pred[3]} {pred[4]}\n")

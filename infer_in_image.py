import cv2
import numpy as np
import torch
from tracknet import BallTrackerNet
import torch.nn.functional as F
from postprocess import postprocess, refine_kps
from homography import get_trans_matrix, refer_kps
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='path to model')
    parser.add_argument('--input_path', type=str, help='path to input image')
    parser.add_argument('--output_path', type=str, help='path to output image')
    parser.add_argument('--use_refine_kps', action='store_true', help='whether to use refine kps postprocessing')
    parser.add_argument('--use_homography', action='store_true', help='whether to use homography postprocessing')
    args = parser.parse_args()

    model = BallTrackerNet(out_channels=15)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    OUTPUT_WIDTH = 640
    OUTPUT_HEIGHT = 360

    image = cv2.imread(args.input_path)
    img = cv2.resize(image, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
    inp = (img.astype(np.float32) / 255.)
    inp = torch.tensor(np.rollaxis(inp, 2, 0))
    inp = inp.unsqueeze(0)

    out = model(inp.float().to(device))[0]

    # [NUM POINTS + 1, HEIGHT, WIDTH]
    pred = F.sigmoid(out).detach().cpu().numpy()


    points = []
    for kps_num in range(14):
        
        heatmap = (pred[kps_num]*255).astype(np.uint8)

        # heatmap is a 1xhxw blackwhtie image, overlay the white on img var

        overlay = img.copy()
        cv2.addWeighted(np.repeat(heatmap[:,:,np.newaxis],3,axis=2), 0.5, overlay, 0.5, 0, overlay)

        cv2.imwrite(f'/home/gabriel/swing-vision/TennisCourtDetector/output/heatmap/{kps_num}.png', overlay)
        x_pred, y_pred = postprocess(heatmap, scale=1,low_thresh=170, max_radius=25)
        
        # draw the point on the image
        overlay = img.copy()
        cv2.circle(overlay, (int(x_pred), int(y_pred)), radius=0, color=(0, 0, 255), thickness=10)

        cv2.imwrite(f'/home/gabriel/swing-vision/TennisCourtDetector/output/circle/{kps_num}_point.png', overlay)

        if args.use_refine_kps and kps_num not in [8, 12, 9] and x_pred and y_pred:
            x_pred, y_pred = refine_kps(image, int(y_pred), int(x_pred))
        points.append((x_pred, y_pred))

    if args.use_homography:
        matrix_trans = get_trans_matrix(points)
        if matrix_trans is not None:
            points = cv2.perspectiveTransform(refer_kps, matrix_trans)
            points = [np.squeeze(x) for x in points]

original_shape = image.shape[:2]
scale_x = original_shape[1] / OUTPUT_WIDTH
scale_y = original_shape[0] / OUTPUT_HEIGHT

for j in range(len(points)):
    if points[j][0] is not None:
        x = int(points[j][0] * scale_x)
        y = int(points[j][1] * scale_y)
        image = cv2.circle(image, (x, y), radius=0, color=(0, 0, 255), thickness=10)


    cv2.imwrite(args.output_path, image)

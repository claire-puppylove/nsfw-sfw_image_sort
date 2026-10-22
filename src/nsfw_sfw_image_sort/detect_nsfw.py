import logging
import argparse
import pprint
import pathlib
from os import PathLike
from typing import Literal
# from nudenet import NudeDetector
# from furry_image_model.model import FurryImageModel
import yaml
import json

####################################
########### GLOBALS INIT ###########
####################################

#############
## Nudenet ##
#############

# tags are detected individually but there's no universal NSFW rating,
# so setting up globals to have condition if individual tags exceed a score to mark them as NSFW

NUDENET_EXPLICIT_LABELS = [
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
]

NUDENET_QUESTIONABLE_LABELS = [
    "FEMALE_GENITALIA_COVERED", # tested, panties pic tests positive
    "FEET_COVERED",
    "ARMPITS_COVERED",
    "BELLY_COVERED",
    "ANUS_COVERED",
    "FEMALE_BREAST_COVERED", # sometimes works on bra, sometimes on clothes
    "BUTTOCKS_COVERED", # tested, panties pic tests positive
    "FEET_EXPOSED",
    "ARMPITS_EXPOSED",
    "BELLY_EXPOSED",
    ]

NUDENET_ALL_LABELS = [
    "FEMALE_GENITALIA_COVERED",
    "FACE_FEMALE",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "FEET_EXPOSED",
    "BELLY_COVERED",
    "FEET_COVERED",
    "ARMPITS_COVERED",
    "ARMPITS_EXPOSED",
    "FACE_MALE",
    "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_COVERED",
    "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
]

###########
## Furry ##
###########

# Has internal tags and nsfw labeling so no globals needed


##############################################
########### DEFAULT DETECTORS INIT ###########
##############################################

def get_detectors(
    nudenet_use:bool,
    furry_model_use:bool,
    nudenet_model_path:pathlib.Path|PathLike|None = None, # if none, it will refer to the inference resolution
    nudenet_inference_resolution:Literal[640,320] = 640,
    nudenet_sess_options=None, # type onnxruntime.SessionOptions | None
    nudenet_providers=None,
    nudenet_run_options=None, # type onnxruntime.RunOptions | None
    furry_model_path:pathlib.Path|PathLike = "./models/convnext_tiny_224_5",
    furry_model_safe_threshold=0.2, # strict
    furry_model_explicit_threshold=0.6, # strict
    ):
    if nudenet_use:
        from nudenet import NudeDetector
        import onnxruntime
    if furry_model_use:
        from furry_image_model.model import FurryImageModel
    if nudenet_use:
        #############
        ## Nudenet ##
        #############
        if not nudenet_model_path:
            if nudenet_inference_resolution==640:
                nudenet_model_path = "./models/640m_nudenet.onnx"
            if nudenet_inference_resolution==320:
                nudenet_model_path = "./models/320n_nudenet.onnx"
            else:
                raise ValueError("specify NudeNet model path")
        # person nude data
        # 320, too pixelated
        nudenet_detector = NudeDetector(
            model_path=pathlib.Path(nudenet_model_path), # 
            inference_resolution=nudenet_inference_resolution, # 640
            sess_options = nudenet_sess_options,
            providers = nudenet_providers,
            run_options = nudenet_run_options,
            # providers=C.get_available_providers() if not nudenet_providers else nudenet_providers,
            )
    if furry_model_use:
        ###########
        ## Furry ##
        ###########
        # lenient nsfw rating
        # furry_model = FurryImageModel(
        #     "./models/convnext_tiny_224_5",
        #     safe_threshold=0.4,
        #     explicit_threshold=0.8)
        #
        # strict nsfw rating
        furry_model = FurryImageModel(
            furry_model_path, # './models/convnext_tiny_224_5'
            safe_threshold=furry_model_safe_threshold, 
            explicit_threshold=furry_model_explicit_threshold)
    if nudenet_use and furry_model_use:
        logging.info("Loaded dict = {'nudenet':nudenet_detector, 'furry_model':furry_model}")
        return {"nudenet":nudenet_detector, "furry_model":furry_model}
    elif nudenet_use:
        logging.info("Loaded single nudenet model")
        return nudenet_detector
    elif furry_model_use:
        logging.info("Loaded single furry model")
        return furry_model
    else:
        logging.warning("No models loaded!")
        return None


##############################
########### DETECT ###########
##############################

def merge_boxes(boxes:list[list[int,int,int,int]]):
    # simple method for merging bounding boxes, no thresholds
    # NudeNet uses openCV (cv2)
    #(xmin,ymin,boxwidth,boxheight)
    # x is left to right
    # y is top to bottom
    leftmost_x = min([b[0] for b in boxes])
    topmost_y = min([b[1] for b in boxes])
    rightmost_x = max([b[0]+b[2] for b in boxes])
    bottommost_y = max([b[1]+b[3] for b in boxes])
    width = rightmost_x - leftmost_x
    height = bottommost_y - topmost_y
    return [leftmost_x,topmost_y,width,height]

def detect(
    image_paths:list[pathlib.Path|PathLike]=[],
    nudenet_use=True,
    nudenet_detector = None, # type NudeDetector
    nudenet_explicit_taglist: list[Literal[tuple(NUDENET_ALL_LABELS)]]=NUDENET_EXPLICIT_LABELS,
    nudenet_explicit_threshold = 0.6,
    nudenet_questionable_taglist: list[Literal[tuple(NUDENET_ALL_LABELS)]]=NUDENET_QUESTIONABLE_LABELS,
    nudenet_questionable_threshold = 0.6,
    nudenet_batch_size = 4,
    furry_model_use=True,
    furry_model_detector = None, # type FurryImageModel
    furry_model_safe_threshold = 0.2,
    furry_model_explicit_threshold = 0.6,
    furry_model_tag_threshold:float = 0.5,
    run_order: tuple[Literal["nudenet","furry_model"],Literal["nudenet","furry_model"]] = ("nudenet","furry_model"),    
    ):
    """
    furry_model_tag_threshold (float): The probablility threshold at which to accept a tag as 
            valid. Defaults to 0.5.
    """
    if len(image_paths)!=0:
        # Models don't accept os.Pathlike, only str
        str_paths = [str(pathlib.Path(i).resolve().absolute()) for i in image_paths]
        out = {p:{} for p in image_paths}
        if nudenet_use:
            from nudenet import NudeDetector
        if furry_model_use:
            from furry_image_model.model import FurryImageModel
        for run_model in run_order:
            if run_model == "nudenet":
                if nudenet_use:
                    nudenet_out = {}
                    if not nudenet_detector:
                        nudenet_detector = get_detectors(nudenet_use=True,furry_model_use=False) # load single, use default nargs
                    nudenet_detections = nudenet_detector.detect_batch(
                        str_paths,
                        batch_size=nudenet_batch_size)
                    for index,det_anim in enumerate(nudenet_detections):
                        cur_path = str_paths[index]
                        for frame_ind,det_frame in enumerate(det_anim['r']):
                            # rate explicit and questionable
                            # NudeNet -> {'box': [13, 0, 258, 173],'class': 'explicit','score': 0.666},
                            # make like NudeNet and calculate UNION of boxes in tag, 
                            # add class:'explicit' or 'questionable',value:max(values in list of tags)
                            explicits = [d for d in det_frame['fr'] if d['class'] in nudenet_explicit_taglist and d['score']>=nudenet_explicit_threshold]
                            if explicits!=[]:
                                explicit_box = merge_boxes([b['box'] for b in explicits])
                                explicit_score = max([s['score'] for s in explicits])
                                explicit_det = {
                                    'box':explicit_box,
                                    'class':'explicit',
                                    'score':explicit_score,
                                }
                                det_anim['r'][frame_ind]['fr'].append(explicit_det)
                            questionables = [d for d in det_frame['fr'] if d['class'] in nudenet_questionable_taglist and d['score']>=nudenet_questionable_threshold]
                            if questionables!=[]:
                                questionable_box = merge_boxes([b['box'] for b in questionables])
                                questionable_score = max([s['score'] for s in questionables])
                                questionable_det = {
                                    'box':questionable_box,
                                    'class':'questionable',
                                    'score':questionable_score,
                                }
                                det_anim['r'][frame_ind]['fr'].append(questionable_det)
                        out[cur_path]["nudenet_detections"] = det_anim
                        logging.info(pprint.pformat(cur_path)+"\n"+pprint.pformat(det_anim,width=160)+"\n")
            if run_model == "furry_model":
                if furry_model_use:
                    furry_model_out = {}
                    if not furry_model_detector:
                        furry_model_detector = get_detectors(nudenet_use=False,furry_model_use=True) # load single, use default nargs
                    furry_model_detections = furry_model_detector.predict_image_tags(*str_paths,t=furry_model_tag_threshold)
                    for index,det in enumerate(furry_model_detections):
                        cur_path = str_paths[index]
                        furry_model_out[cur_path] = det
                        logging.info(pprint.pformat(cur_path)+"\n"+pprint.pformat(det,width=160)+"\n")
                    out[cur_path]["furry_model_detections"]=furry_model_out
        return out
    else:
        logging.warning("No images or animations loaded.")
        return None

def main():
    parser = argparse.ArgumentParser()
    # INPUT PATHS
    parser.add_argument("-i", "--input", type=str, default="./test_images/", help="single file, single folder path or glob")
    # add output file
    parser.add_argument("-o", "--output", type=str, default="./outputs/detection.json", help="single file")
    # get_detectors()
    parser.add_argument('--nudenet_use', action='store_true')
    parser.add_argument('--no-nudenet_use', dest='nudenet_use', action='store_false')
    parser.set_defaults(nudenet_use=True)
    parser.add_argument("--nudenet_model_path", type=str, default=None)
    parser.add_argument("--nudenet_inference_resolution", type=int, default=640, help="640 or 320 only")
    parser.add_argument("--nudenet_gpu", action='store_true')
    parser.add_argument("--nudenet_cpu", dest='nudenet_gpu', action='store_false')
    parser.set_defaults(nudenet_gpu=True)
    parser.add_argument('--furry_model_use', action='store_true')
    parser.add_argument('--no-furry_model_use', dest='furry_model_use', action='store_false')
    parser.set_defaults(furry_model_use=True)
    parser.add_argument("--furry_model_path", type=str, default="./models/convnext_tiny_224_5")
    parser.add_argument("--furry_model_safe_threshold", type=float, default=0.2, help="Strict ~0.2, Lenient ~0.4")
    parser.add_argument("--furry_model_explicit_threshold", type=float, default=0.6, help="Strict ~0.6, Lenient ~0.8")
    # detect()
    parser.add_argument('--nudenet_explicit_taglist', nargs='*', help='explicit tags separated by spaces, empty or omit arg for default NUDENET_EXPLICIT_LABELS', default=NUDENET_EXPLICIT_LABELS)
    parser.add_argument("--nudenet_explicit_threshold", type=float, default=0.6, help="Strict ~0.6, Lenient ~0.8")
    parser.add_argument('--nudenet_questionable_taglist', nargs='*', help='questionable tags separated by spaces, empty or omit arg for default NUDENET_QUESTIONABLE_LABELS', default=NUDENET_QUESTIONABLE_LABELS)
    parser.add_argument("--nudenet_questionable_threshold", type=float, default=0.6, help="Strict ~0.6, Lenient ~0.8")
    parser.add_argument("--nudenet_batch_size", type=int, default=4)
    parser.add_argument("--furry_model_tag_threshold", type=float, default=0.5, help="Probability threshold of tag existing")
    parser.add_argument("--run_order", type=str, default="nf", help="nf='nudenet,furry_model' or fn='furry_model,nudenet'")
    parser.add_argument('-d', '--debug',help="Print lots of debugging statements",action="store_const", dest="loglevel", const=logging.DEBUG,default=logging.WARNING)
    parser.add_argument('-v', '--verbose',help="Be verbose",action="store_const", dest="loglevel", const=logging.INFO)
    parser.add_argument('--yaml',help="output to yaml",action="store_const", dest="output_type", const="yaml")
    parser.add_argument('--json',help="output to json (pprint formatted)",action="store_const", dest="output_type", const="json")
    parser.add_argument("-n", "--image_batch_size", type=int, default=100)
    parser.set_defaults(output_type="json")
    # parse args
    args = parser.parse_args()
    # logging
    logging.basicConfig(level=args.loglevel)
    logging.info("Args:"+str(args)+"\n")
    # Parse input paths
    if pathlib.Path(args.input).is_dir():
        input_paths = [str(pd.resolve().absolute()) for pd in pathlib.Path(args.input).iterdir()]
    else:
        input_paths = [str(pathlib.Path(p).resolve().absolute()) for p in glob.iglob(args.input)]
    # initialize detectors
    # default
    nudenet_sess_options = None
    nudenet_run_options = None
    if args.nudenet_gpu:
        nudenet_providers = ["CUDAExecutionProvider"]
    else:
        nudenet_providers = None
    detectors = get_detectors(
        nudenet_use = args.nudenet_use,
        furry_model_use = args.furry_model_use,
        nudenet_model_path = args.nudenet_model_path,
        nudenet_inference_resolution = args.nudenet_inference_resolution,
        nudenet_sess_options = nudenet_sess_options,
        nudenet_providers = nudenet_providers,
        nudenet_run_options = nudenet_run_options,
        furry_model_path = args.furry_model_path,
        furry_model_safe_threshold = args.furry_model_safe_threshold,
        furry_model_explicit_threshold = args.furry_model_explicit_threshold,
        )
    nudenet_detector = None
    furry_model_detector = None
    if args.nudenet_use and args.furry_model_use:
        nudenet_detector = detectors['nudenet']
        furry_model_detector = detectors['furry_model']
    elif args.nudenet_use:
        nudenet_detector = detectors
    elif args.furry_model_use:
        furry_model_detector = detectors
    elif not args.nudenet_use and not args.furry_model_use:
        raise ValueError("No models initialized")
    # detect
    if args.run_order == "nf":
        run_order = ("nudenet","furry_model")
    elif args.run_order == "fn":
        run_order = ("furry_model","nudenet")
    else:
        raise ValueError("Run order must be nf or fn")
    # run image batches
    cur_batch = 1
    total_batches = len(range(0, len(input_paths), args.image_batch_size))
    out = {}
    for i in range(0, len(input_paths), args.image_batch_size):
        image_batch_input_paths = input_paths[i:i+args.image_batch_size]
        logging.info(f"Running image batch {cur_batch} of {total_batches}")
        image_batch_out = detect(
            image_paths = image_batch_input_paths,
            nudenet_use = args.nudenet_use,
            nudenet_detector = nudenet_detector,
            nudenet_explicit_taglist = args.nudenet_explicit_taglist,
            nudenet_explicit_threshold = args.nudenet_explicit_threshold,
            nudenet_questionable_taglist = args.nudenet_questionable_taglist,
            nudenet_questionable_threshold = args.nudenet_questionable_threshold,
            nudenet_batch_size = args.nudenet_batch_size,
            furry_model_use = args.furry_model_use,
            furry_model_detector = furry_model_detector,
            furry_model_safe_threshold = args.furry_model_safe_threshold,
            furry_model_explicit_threshold = args.furry_model_explicit_threshold,
            furry_model_tag_threshold = args.furry_model_tag_threshold,
            run_order = run_order,
            )
        out.update(image_batch_out)
        cur_batch+=1
    if args.output_type == "json":
        out_str = json.dumps(out, indent=2)
    elif args.output_type == "yaml":
        out_str = yaml.dump(out)    
    with open(args.output,"w") as f:
        f.write(out_str)



if __name__ == "__main__":
    main()
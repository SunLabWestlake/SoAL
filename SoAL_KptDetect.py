# -*- coding: utf-8 -*-

"""
Main: Keypoint detection
Author: Jing Ning @ SunLab

update log:
V1.0 20200904 support HRNet
V0.8 20200410 multi-image-parallel-processing
V0.5 20190903 multi-task and add GUI
V0.3 20190613 track and extract video
V0.1 20181221 centralization and track

Usage:
>> SoAL_KptDetect.py all

SoAL processing stages:
1. pre-processing (standalone)
    frame_align (optional)
    input_meta_info
        input_exp_info
        input_roi_info
        input_bg_info
2. region segmentation
    background subtraction
    region segmentation
    centralization
3. key points detection
4. write results


directory structure:

 #: file property
 {}: input item
 *: copy from above

exp(D:\exp)
    |-log.csv
    |   |-exp_date, start, file, duration, temperature, female_days, male_geno_days
    |-exp.xlsx
    |-20190807_140000_A
        |-.state (init|running|finish)
        |-20190807_140000_A.avi (0_frame_align)
            |-#Duration, #FPS
        |-20190807_140000_A_bg.bmp (0_calc_bg)
        |-20190807_140000_A.log
        |-20190807_140000_A_config.json (0_input_meta_info)
            |-file, total_frame, width, height, camera, start, end, duration, FPS
            |-VERSION, ROUND_ARENA, MODEL_FOLDER
            |-{SCALE}, {temperature}, {exp_date}, {female_date}, female_days
            |-{GRAY_THRESHOLD}
            |-ROI
                |-idx, fly_id, {roi(2x2)}, {info}, {male_geno}, {male_date}, male_days
                |-...
        |-0
            |-20190807_140000_A_0_kpt.csv
               |-frame, time, reg_n, area, pos:x, pos:y, ori, e_maj, e_min, point:xs, point:ys...
               |-...
            |-20190807_140000_A_0_config.json
                |-*INFO
                |-*ROI
                    |-idx, fly_id, {roi(2x2)}, {info}, {male_geno}, {male_date}, male_days

"""

from SoAL_Constants import IMG_QUEUE_CAPACITY, PRED_PROCESS_NUM, VIDEO_TODO_DIR, MAX_TASK

# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def soal_main():
    print("[Main] start...")
    from multiprocessing import Process, Queue
    from SoAL_PartsPrediction import PredictParts
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    pred_q_l = []
    for i in range(PRED_PROCESS_NUM):
        pred_q = Queue(maxsize=IMG_QUEUE_CAPACITY)
        pred_p = Process(target=PredictParts.process, args=(pred_q, i))
        pred_p.start()
        pred_q_l.append(pred_q)

    from SoAL_BodyDetection import RegionSeg
    from datetime import datetime
    import time
    last_task_n = 0
    task_id = 0
    task_finish = 0
    total_task = 0
    while True:
        print("[Main] check dir... (%d tasks running) (%d/%d tasks finished)" % (last_task_n, task_finish, total_task))
        task_n = 0
        task_finish = 0
        total_task = 0
        vl = os.listdir(VIDEO_TODO_DIR)
        vl.sort()
        for video_dir in vl:
            video_dir_path = os.path.join(VIDEO_TODO_DIR, video_dir)
            if os.path.isdir(video_dir_path):
                state_file = os.path.join(video_dir_path, ".state")
                total_task += 1
                if os.path.exists(state_file):
                    try:
                        f = open(state_file, "r")
                    except Exception:
                        continue
                    state = f.readline().replace("\n", "")
                    if state == "init":
                        if last_task_n >= MAX_TASK or task_n >= MAX_TASK:
                            f.close()
                            continue
                        print("[Main]: init task: %s" % video_dir)
                        video = os.path.join(video_dir_path, video_dir + ".avi")
                        if not os.path.exists(video):
                            print("[Main]: error!!! %s not found" % video)
                        pred_q = pred_q_l[task_id % PRED_PROCESS_NUM]
                        pred_q.put([task_id, video])

                        rs_p = Process(target=RegionSeg.process, args=(video, task_id, pred_q))
                        rs_p.start()

                        f.close()
                        f = open(state_file, "w")
                        f.write("running\n" + "%d %s\n" % (task_id, datetime.strftime(datetime.now(), "%Y%m%d %H:%M:%S")))

                        task_id += 1
                        task_n += 1
                        # break
                    elif state == "running":
                        task_n += 1
                    elif state.startswith("finish"):
                        task_finish += 1

                    f.close()

        last_task_n = task_n
        if task_finish >= total_task:
            break
        time.sleep(15)
    print("[Main]: all finished")

def write_init_file(file_l):
    for file in file_l:
        f = open(file, "w")
        f.write("init\n")
        f.close()

if __name__ == '__main__':
    file_l = []
    import sys, os
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if len(path) <= 3:
            if path != "all":
                MAX_TASK = int(path)
            for video_dir in os.listdir(VIDEO_TODO_DIR):
                video_dir_path = os.path.join(VIDEO_TODO_DIR, video_dir)
                if os.path.isdir(video_dir_path):
                    file_l.append(os.path.join(video_dir_path, ".state"))
        else:
            if not os.path.isdir(path):
                path = os.path.dirname(path)
            file_l.append(os.path.join(path, ".state"))
    write_init_file(file_l)
    soal_main()

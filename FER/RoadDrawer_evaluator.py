
from ResidualMaskingNetwork.rmn import RMN
from matplotlib import pyplot as plt
import argparse
import cv2
import time
import threading
from tkinter import *
import sys

class RoadDrawer_Evaluator:
    def __init__(self, save_path) -> None:

        self.save_path = save_path

        self.m = RMN()
        self.cap = cv2.VideoCapture(0)
        
        self.initialize()
        
        self.root = Tk()
        self.root.title("Road Drawer evaluator")
        self.root.geometry("300x200+100+100")
        self.root.resizable(True, True)
        
        start_button = Button(self.root, text="start", width=10, command=self.start_evaluation)
        stop_button = Button(self.root, text="stop", width=10, command=self.stop_evaluation)
        start_button.pack(side="left", anchor="n")
        stop_button.pack(side="right", anchor="n")
        
        self.time_label = Label(self.root, text="0.0")
        self.time_label.pack(side="top")
        
        self.root.mainloop()
        
    def initialize(self):
        self.time_list = []
        self.prob_listoflist = [[],[],[],[],[],[],[]]
        self.is_running = False
        self.fer_class = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.avg_proba_list = [{self.fer_class[0]:0},{self.fer_class[1]:0},{self.fer_class[2]:0},{self.fer_class[3]:0},{self.fer_class[4]:0},{self.fer_class[5]:0},{self.fer_class[6]:0}]
        self.fer_count = [0,0,0,0,0,0,0]
        self.total_frame = 1
        

    def plot_fer_prob(self, t, tmp_prob_list : list):     # tmp_prob_list only contains probabilities
        self.time_list.append(t)
        for k in range(7):
            self.prob_listoflist[k].append(tmp_prob_list[k])
            
    def stop_evaluation(self):
        self.is_running = False
        
        print("----------------------")
        print(f"building time : {time.time() - self.start_time}")
        sum_check = 0
        for k in range(7):
            plt.plot(self.time_list,self.prob_listoflist[k], label=self.fer_class[k])
            print(f'{self.fer_class[k]} ratio : {self.fer_count[k]/(self.total_frame-1)}')
            sum_check += self.fer_count[k]
            
        print("sum check must be 1.0. got", sum_check/(self.total_frame-1))
        print("----------------------")
        
        plt.legend()
        plt.ylabel("probability")
        plt.xlabel("time(sec)")
        print(self.save_path)
        plt.savefig(self.save_path)
        # plt.savefig("FER_graphs/test.png")
        
        self.root.destroy()
        sys.exit()
            
    def start_evaluation(self):
        self.is_running = True
        self.start_time = time.time()
        self.run_thread = threading.Thread(target=self.run)
        self.run_thread.start()
        
    def run(self):
        while self.is_running:
            ret, frame = self.cap.read()
            
            results = self.m.detect_emotion_for_single_frame(frame)
            
            tmp_prob_list = []
            if len(results):
                self.fer_count[self.fer_class.index(results[0]['emo_label'])] += 1
                proba_list = results[0]['proba_list']
                for i, dic in enumerate(proba_list):
                    label_name = self.fer_class[i]
                    tmp_prob = dic[label_name]
                    
                    self.avg_proba_list[i][label_name] = ((self.total_frame-1)*self.avg_proba_list[i][label_name] + tmp_prob)/self.total_frame
                    tmp_prob_list.append(self.avg_proba_list[i][label_name])
                    
                self.plot_fer_prob(time.time() - self.start_time, tmp_prob_list)
                
                
                self.total_frame+=1
                                
            image = self.m.draw(frame, results)
            cv2.imshow("emo", image)
            ch = cv2.waitKey(400)
            if ch == 27 or ch == "q" or ch == "Q":
                break
        
            self.time_label.config(text='%0.2f s' % (time.time() - self.start_time))
            
        self.cap.release()
        cv2.destroyAllWindows()
            
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--save-path', default="FER_graphs/test.png", help="path to save FER probability graphs")
    args = parser.parse_args()
    run = RoadDrawer_Evaluator(args.save_path)
    
    
    
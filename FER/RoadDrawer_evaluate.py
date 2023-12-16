
from ResidualMaskingNetwork.rmn import RMN
from matplotlib import pyplot as plt
import cv2
import time

time_list = []
prob_listoflist = [[],[],[],[],[],[],[]]
def plot_fer_prob(t, tmp_prob_list : list):     # tmp_prob_list only contains probabilities
    time_list.append(t)
    for k in range(7):
        prob_listoflist[k].append(tmp_prob_list[k])
        
cap = cv2.VideoCapture(0)
m = RMN()

fer_class = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
avg_proba_list = [{fer_class[0]:0},{fer_class[1]:0},{fer_class[2]:0},{fer_class[3]:0},{fer_class[4]:0},{fer_class[5]:0},{fer_class[6]:0}]
fer_count = [0,0,0,0,0,0,0]
n = 1
start_time = time.time()
print(start_time)
while True:
    ret, frame = cap.read()
    
    results = m.detect_emotion_for_single_frame(frame)
    
    tmp_prob_list = []
    if len(results):
        fer_count[fer_class.index(results[0]['emo_label'])] += 1
        proba_list = results[0]['proba_list']
        for i, dic in enumerate(proba_list):
            label_name = fer_class[i]
            tmp_prob = dic[label_name]
            
            avg_proba_list[i][label_name] = ((n-1)*avg_proba_list[i][label_name] + tmp_prob)/n
            tmp_prob_list.append(avg_proba_list[i][label_name])
            
        plot_fer_prob(time.time() - start_time, tmp_prob_list)
        
        n+=1
                        
    image = m.draw(frame, results)
    cv2.imshow("emo", image)
    ch = cv2.waitKey(400)
    if ch == 27 or ch == "q" or ch == "Q":
        break
    
    
    
print("----------------------")
print(f"building time : {time.time() - start_time}")
sum_check = 0
for k in range(7):
    plt.plot(time_list,prob_listoflist[k], label=fer_class[k])
    print(f'{fer_class[k]} ratio : {fer_count[k]/n}')
    sum_check += fer_count[k]
    
# print("sum check must be 1.0. got", sum_check/n)
print("----------------------")
    
plt.legend()
plt.ylabel("probability")
plt.xlabel("time(sec)")
plt.savefig("FER_graph.png")
cv2.destroyAllWindows()
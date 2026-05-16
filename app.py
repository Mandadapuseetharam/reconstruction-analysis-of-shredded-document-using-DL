import tkinter as tk
from tkinter import filedialog
import numpy as np, os, cv2, pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.layers import MaxPooling2D, Dense, Flatten, Convolution2D
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

BG,PAN,CARD,ACC,CYN,GRN,RED,TXT,MUT,BRD = (
    "#0a0a0d","#0f0f15","#0a0a0d","#f59e0b","#38bdf8","#22c55e",
    "#ef4444","#f1f5f9","#64748b","#1e1e26")
STEP_NAMES = ["Upload","Preprocess","Split","Run CNN","Reconstruct"]

filename=cnn_model=X=Y=None
X_train=X_test=y_train=y_test=order=labels=None
cur_step = [0]

root = tk.Tk()
root.title("DocRecon — Forensic Document Intelligence")
root.geometry("1280x800"); root.configure(bg=BG); root.resizable(True,True)

def log(msg, tag="n"):
    out.configure(state="normal"); out.insert(tk.END, msg+"\n", tag)
    out.see(tk.END); out.configure(state="disabled")
def log_clear():
    out.configure(state="normal"); out.delete("1.0",tk.END); out.configure(state="disabled")
def set_status(msg, col=GRN):
    sv.set(msg); sl.configure(fg=col)
def getLabel(name):
    return next((i for i,l in enumerate(labels) if l==name),-1)

def draw_pipeline(event=None):
    sc.delete("all"); W=sc.winfo_width() or 1280; H=72; n=len(STEP_NAMES)
    cell=W//n
    for i,s in enumerate(STEP_NAMES):
        cx=cell*i+cell//2; cy=H//2-6
        done=(i<cur_step[0]); active=(i==cur_step[0])
        if i<n-1:
            lc=GRN if done else (ACC if active else BRD)
            sc.create_line(cx+24,cy,cx+cell-24,cy,fill=lc,width=2)
        r=16 if active else 13
        fill=ACC if active else (PAN if not done else "#0d1f12")
        outline=ACC if active else (GRN if done else BRD)
        sc.create_oval(cx-r,cy-r,cx+r,cy+r,fill=fill,outline=outline,width=2)
        txt="✓" if done else str(i+1)
        fc="#0a0a0d" if active else (GRN if done else MUT)
        sc.create_text(cx,cy,text=txt,fill=fc,font=("Segoe UI",9,"bold"))
        lc2=ACC if active else (GRN if done else MUT)
        sc.create_text(cx,cy+24,text=s,fill=lc2,font=("Segoe UI",8,"bold" if active else "normal"))

def advance(n):
    cur_step[0]=n; draw_pipeline()

# ── Business logic ─────────────────────────────────────────────────────────
def upload():
    global filename,labels,order,X,Y
    labels,order,X,Y=[],[],[],[]
    d=filedialog.askdirectory(initialdir=".")
    if not d: return
    filename=d; log_clear(); advance(0)
    for _,_,files in os.walk(filename+"/gt"):
        for f in files:
            if f.strip() not in labels: labels.append(f.strip())
    if os.path.exists('model/X.txt.npy'):
        X=np.load('model/X.txt.npy'); Y=np.load('model/Y.txt.npy')
    else:
        for r2,_,files in os.walk(filename+"/stripes"):
            for f in files:
                if f=="order.txt":
                    for line in open(r2+"/"+f,"rb").read().decode().split("\n"):
                        fn=line.strip()
                        if fn: order.append(r2+"/"+fn+".png")
        for path in order:
            lbl=os.path.basename(os.path.dirname(path)).split("_")
            if os.path.exists("Dataset/gt/"+lbl[0]+".png"):
                shred=cv2.imread(path)
                X.append(cv2.resize(shred,(32,32))); Y.append(getLabel(lbl[0]+".png"))
        X,Y=np.asarray(X),np.asarray(Y)
        np.save('model/X.txt',X); np.save('model/Y.txt',Y)
    log(f"  DATASET  /  {filename}","acc"); log(f"  Strips loaded  →  {X.shape[0]} images","suc")
    set_status("Dataset loaded"); advance(1)

def preprocess():
    global X,Y; log_clear()
    X=X.astype('float32')/255
    idx=np.arange(X.shape[0]); np.random.shuffle(idx)
    X,Y=X[idx],Y[idx]; Y=to_categorical(Y)
    log("  PREPROCESS  /  Shuffle + Normalise","acc"); log("  Complete — images ready","suc")
    set_status("Preprocessed"); advance(2)

def trainSplit():
    global X_train,X_test,y_train,y_test; log_clear()
    X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)
    log(f"  SPLIT  /  Total records : {X.shape[0]}","acc")
    log(f"  Train 80%  →  {X_train.shape[0]}","inf"); log(f"  Test  20%  →  {X_test.shape[0]}","suc")
    set_status("Dataset split"); advance(3)

def calcMetrics(algo,yt,yp):
    a=accuracy_score(yt,yp)*100; p=precision_score(yt,yp,average='macro')*100
    r=recall_score(yt,yp,average='macro')*100; f=f1_score(yt,yp,average='macro')*100
    log(f"\n  ┌── {algo}","mut")
    log(f"  │  Accuracy   {a:.2f}%","suc"); log(f"  │  Precision  {p:.2f}%","inf")
    log(f"  │  Recall     {r:.2f}%","inf"); log(f"  └─ F-Score   {f:.2f}%","inf")

def runCNN():
    global cnn_model; log_clear()
    cnn_model=Sequential([
        Convolution2D(32,(3,3),input_shape=(X_train.shape[1],X_train.shape[2],X_train.shape[3]),activation='relu'),
        MaxPooling2D((2,2)),Convolution2D(32,(3,3),activation='relu'),MaxPooling2D((2,2)),
        Flatten(),Dense(256,activation='relu'),Dense(y_train.shape[1],activation='softmax')])
    cnn_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    log("  CNN  /  Model compiled","acc")
    if not os.path.exists("model/cnn_weights.hdf5"):
        ckpt=ModelCheckpoint('model/cnn_weights.hdf5',verbose=1,save_best_only=True)
        hist=cnn_model.fit(X_train,y_train,batch_size=32,epochs=30,
                           validation_data=(X_test,y_test),callbacks=[ckpt],verbose=1)
        pickle.dump(hist.history,open('model/cnn_history.pckl','wb'))
    else:
        cnn_model.load_weights("model/cnn_weights.hdf5"); log("  Weights loaded from cache","suc")
    pred=np.argmax(cnn_model.predict(X_test),axis=1)
    calcMetrics("CNN Document Reconstruction",np.argmax(y_test,axis=1),pred)
    set_status("CNN trained & evaluated"); advance(4)

def reconstruct():
    global cnn_model
    d=filedialog.askdirectory(initialdir="testData")
    if not d: return
    log_clear(); test,shred=[],[]
    for r2,_,files in os.walk(d):
        for f in files:
            img=cv2.imread(r2+"/"+f); shred.append(img); test.append(cv2.resize(img,(32,32)))
    test=np.asarray(test).astype('float32')/255
    pred=np.argmax(cnn_model.predict(test),axis=1); best=np.argmax(np.bincount(pred))
    log(f"  RECONSTRUCT  /  {test.shape[0]} pieces analysed","acc")
    log(f"  Best match  →  {labels[best]}","suc")
    cv2.imshow("Shredded Documents",cv2.hconcat(shred))
    cv2.imshow("Reconstructed Document",cv2.imread("Dataset/gt/"+labels[best]))
    cv2.waitKey(0); set_status("Reconstruction complete")

# ── UI Build ───────────────────────────────────────────────────────────────
hdr=tk.Frame(root,bg=PAN,height=58); hdr.pack(fill=tk.X); hdr.pack_propagate(False)
tk.Label(hdr,text="◈",font=("Segoe UI",20,"bold"),bg=PAN,fg=ACC).pack(side=tk.LEFT,padx=(18,6),pady=10)
tk.Label(hdr,text="DocRecon",font=("Segoe UI",15,"bold"),bg=PAN,fg=TXT).pack(side=tk.LEFT,pady=10)
tk.Label(hdr,text="  ·  Forensic Document Reconstruction  ·  Deep Learning",
         font=("Segoe UI",10),bg=PAN,fg=MUT).pack(side=tk.LEFT,pady=14)
tk.Frame(root,bg=ACC,height=2).pack(fill=tk.X)

sc=tk.Canvas(root,bg=BG,height=72,highlightthickness=0); sc.pack(fill=tk.X)
sc.bind("<Configure>",draw_pipeline)
tk.Frame(root,bg=BRD,height=1).pack(fill=tk.X)

out_wrap=tk.Frame(root,bg=CARD); out_wrap.pack(fill=tk.BOTH,expand=True,padx=14,pady=10)
out=tk.Text(out_wrap,bg=CARD,fg=TXT,font=("Lucida Console",11),bd=0,padx=14,pady=10,
            state="disabled",wrap=tk.WORD,insertbackground=ACC,selectbackground="#2d2d40")
osb=tk.Scrollbar(out_wrap,command=out.yview,bg=PAN,troughcolor=BG,bd=0,width=10)
out.configure(yscrollcommand=osb.set)
osb.pack(side=tk.RIGHT,fill=tk.Y); out.pack(fill=tk.BOTH,expand=True)
for tag,col in [("acc",ACC),("suc",GRN),("inf",CYN),("mut",MUT),("n",TXT)]:
    out.tag_configure(tag,foreground=col)

tk.Frame(root,bg=BRD,height=1).pack(fill=tk.X)

btn_bar=tk.Frame(root,bg=PAN,height=56); btn_bar.pack(fill=tk.X,side=tk.BOTTOM); btn_bar.pack_propagate(False)
ACTIONS=[("⬆  Upload Dataset",upload,ACC,"#0a0a0d"),("⚙  Preprocess",preprocess,"#3b82f6",TXT),
         ("✂  Split",trainSplit,"#8b5cf6",TXT),("▶  Run CNN",runCNN,"#ec4899",TXT),
         ("◎  Reconstruct",reconstruct,GRN,"#0a0a0d"),("✕  Exit",root.destroy,"#1a1a24",RED)]

def mk(parent,label,cmd,bg,fg):
    f=tk.Frame(parent,bg=bg,cursor="hand2"); f.pack(side=tk.LEFT,padx=5,pady=10)
    l=tk.Label(f,text=label,bg=bg,fg=fg,font=("Segoe UI",10,"bold"),padx=14,pady=5,cursor="hand2"); l.pack()
    def ei(e): f.configure(bg=TXT); l.configure(bg=TXT,fg="#0a0a0d")
    def eo(e): f.configure(bg=bg);  l.configure(bg=bg,fg=fg)
    for w in (f,l):
        w.bind("<Enter>",ei); w.bind("<Leave>",eo); w.bind("<Button-1>",lambda e,c=cmd:c())

for lbl,cmd,bg,fg in ACTIONS: mk(btn_bar,lbl,cmd,bg,fg)

sf=tk.Frame(root,bg="#07070a",height=26); sf.pack(fill=tk.X,side=tk.BOTTOM); sf.pack_propagate(False)
sv=tk.StringVar(value="Ready")
sl=tk.Label(sf,textvariable=sv,bg="#07070a",fg=MUT,font=("Segoe UI",9),anchor="w",padx=14)
sl.pack(side=tk.LEFT,fill=tk.Y)

log("  Welcome to DocRecon — Forensic Document Reconstruction System","acc")
log("  Follow the 5 pipeline steps in order  ·  left → right","mut")

root.mainloop()

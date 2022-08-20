import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score as R2

def opCSV(path = ""):
    t,i=[],[]

    with open(path,'r') as file:
        reader = csv.reader(file,dialect='excel',delimiter='\t')
        for row in reader:
            t.append(row[0])
            i.append(row[1])
        t = t[1:]
        i = i[1:]
        t = [float(j) for j in t]
        i = [float(j) for j in i]
    return np.array(t),np.array(i)

#%% Apertura CSV
tb,ib = opCSV("Eblanco.csv")
tr, ir = opCSV("Erojo_Zadj.csv")
trv,irv = opCSV("Erojo_verde.csv")

#%% Tabla
datos = pd.DataFrame({"tiempo (seg)":tb,"Electrodo 1 (mA)":irv,"Electrodo 2 (mA)":ib})
print("                        Corriente (mA)")
print(datos)

#%%Regresiones

def adjLIN(x,y): #y = ax + b
    return np.polyfit(x, y, 1)

def adjCuad(x,y):
    return np.polyfit(x,y,2)

def fcuad(a,b,c,x):
    return a*x**2+b*x+c

def flin(a,b,x):
    return a*x+b

def adjEXP(x,y): #y = ae^bx
    ln_y = np.log(y)
    coef = np.polyfit(x, ln_y, 1)
    #return a,b
    return np.exp(coef[1]), coef[0]

def fexp(a,b,x):
    return a*np.exp(b*x)

def adjLOG(x,y): #y = alog(x+1)+b
    ln_x = np.log(x+1)
    coef = np.polyfit(ln_x, y, 1)
    #return a, b
    return coef[0], coef[1]

def fln(a,b,x):
    return a*np.log(x+1)+ b

def adjPOT(x,y): #y = a(x+1)^(-b)
    ln_x = np.log(x+1)
    ln_y = np.log(y)
    coef = np.polyfit(ln_x,ln_y,1)
    #return a,b
    return np.exp(coef[1]), coef[0]

def fpot(a,b,x):
    return a*(x+1)**b

#%% Mejor ajuste
def BestAdj(x,y):
    keys = ["lin","exp", "ln", "pot","cuad"]
    reg = {"lin": adjLIN ,"exp": adjEXP, "ln": adjLOG, "pot":adjPOT}
    f = {"lin": flin,"exp": fexp, "ln": fln, "pot": fpot}
    r2 = []
    
    for key in keys[:-1]:
        ac = reg[key] #cálculo coeficientes
        fc = f[key] #función
        a,b = ac(x,y)
        pred = fc(a,b,x)
        r2.append(R2(y,pred))
    
    a,b,c = adjCuad(x, y)
    pred = fcuad(a,b,c,x)
    r2.append(R2(y,pred))
    
    Rcuad = max(r2)
    best = keys[r2.index(Rcuad)]
    
    try:
        ac = reg[best]
        a,b = ac(x,y)
    except Exception:
        a,b,c = adjCuad(x, y)
    
    if best == "lin":
        label = "${:.3f}t+{:.3f}$".format(a,b)
    elif best == "exp":
        label = "${:.3f}e^({:.3f}t)$".format(a,b)
    elif best == "ln":
        label = "${:.3f}\ln(t+1)+{:.3f}$".format(a,b)
    elif best == "pot":
        label = "${:.3f}(x+1)^{:.3f}$".format(a,b)
    else:
        label = "${:.3f}x^2+{:.3f}x+{:.3f}$".format(a,b,c)
    
    if best == "cuad":
        t = np.linspace(0,x[-1],1000)
        i = fcuad(a,b,c,t)
    else:
        fc = f[best]
        t = np.linspace(0,x[-1],1000)
        i = fc(a,b,t)
    return t, i, label, Rcuad

#%%Graficas
def graf(t,i, titulo):
    plt.figure()
    tpred,ipred,l,Rcuad= BestAdj(t, i)
    plt.plot(t,i,"o")
    plt.plot(tpred,ipred,label = l)
    plt.title(titulo)
    plt.grid(linestyle="--")
    plt.legend(title = "Recta de ajuste con $R^2={:.3f}$".format(Rcuad))
    plt.xlabel("Tiempo (seg)")
    plt.ylabel("Corriente (mA)")
    plt.show()

graf(trv,irv,"Electrodo 1: color rojo con etiqueta verde, clorado a 1.5 v")
graf(tb,ib,"Electrodo 2: color blanco, clorado a 3.3 v")
graf(tr,ir,"Electrodo 3: color rojo, clorado a 3.3 v")

#%% Todos los datos

plt.figure()
plt.title("Comparación de los datos reales de los tres electrodos")
plt.plot(trv, irv, label = "Electrodo 1 (1.5 v)")
plt.plot(tb,ib, label = "Electrodo 2 (3.3 v)")
plt.plot(tr,ir, label = "Electrodo 3 (3.3 v)")
plt.grid(linestyle = "--")
plt.xlabel("Tiempo (seg)")
plt.ylabel("Corriente (mA)")
plt.legend()
plt.show()


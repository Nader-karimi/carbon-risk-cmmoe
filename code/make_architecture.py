from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
ROOT=Path(__file__).resolve().parents[1]
F=ROOT/'figures';F.mkdir(exist_ok=True)
fig,ax=plt.subplots(figsize=(10,5.8));ax.set_xlim(0,10);ax.set_ylim(0,6);ax.axis('off')
def box(x,y,w,h,text):
    p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04,rounding_size=0.08',fill=False,linewidth=1.2)
    ax.add_patch(p);ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=9);return (x,y,w,h)
def arrow(a,b):
    x1=a[0]+a[2];y1=a[1]+a[3]/2;x2=b[0];y2=b[1]+b[3]/2
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='->',mutation_scale=12,linewidth=1.0))
inputb=box(.25,2.5,1.35,1.0,'Causal EUA\nOHLCV history')
meanb=box(2.0,4.1,1.75,1.0,'Mean experts\nNo-change / AR(5) / ML')
varb=box(2.0,1.0,1.75,1.35,'Variance experts\nLog-SV / HAR-RV /\nEWMA / multiscale ML')
meanfs=box(4.25,4.1,1.55,1.0,'Fixed-share\nmean routing')
varfs=box(4.25,1.15,1.55,1.0,'QLIKE fixed-share\nvariance routing')
meanout=box(6.3,4.1,1.3,1.0,'Conditional mean\n$\\hat\\mu_{t+1|t}$')
varout=box(6.3,1.15,1.3,1.0,'Conditional variance\n$\\hat V_{t+1|t}$')
dist=box(8.1,2.45,1.55,1.25,'Student-$t$ head\n+ conformal\ncalibration')
arrow(inputb,meanb);arrow(inputb,varb);arrow(meanb,meanfs);arrow(varb,varfs);arrow(meanfs,meanout);arrow(varfs,varout);arrow(meanout,dist);arrow(varout,dist)
ax.text(8.88,1.45,'Outputs: density, intervals,\nVaR and Expected Shortfall',ha='center',va='center',fontsize=9)
fig.tight_layout();fig.savefig(F/'cmd_moe_architecture.png',dpi=240,bbox_inches='tight');plt.close(fig)

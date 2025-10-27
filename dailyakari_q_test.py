import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import random

# === パズル定義 ===
rows, cols = 5, 5
grid = np.zeros((rows, cols), dtype=int)
numbered_walls = {(0,1):0, (1,3):1, (3,1):2}
plain_walls = []
for (r, c), v in numbered_walls.items(): grid[r, c] = 10
for r, c in plain_walls: grid[r, c] = -1
original_numbers = numbered_walls.copy()

# === 光計算・残り電球 ===
def compute_shine(lights):
    shines = np.zeros_like(grid, dtype=bool)
    for r in range(rows):
        for c in range(cols):
            if lights[r, c] == 2:
                shines[r, c] = True
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    rr, cc = r+dr, c+dc
                    while 0 <= rr < rows and 0 <= cc < cols:
                        if grid[rr, cc] == -1 or grid[rr, cc] == 10: break
                        shines[rr, cc] = True
                        rr += dr; cc += dc
    return shines

def remaining_number(lights):
    remain = {}
    for (r,c), v in original_numbers.items():
        cnt = 0
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            rr, cc = r+dr, c+dc
            if 0<=rr<rows and 0<=cc<cols and lights[rr,cc]==2: cnt+=1
        remain[(r,c)] = max(v-cnt,0)
    return remain

def check_clear(lights):
    shines = compute_shine(lights)
    for r in range(rows):
        for c in range(cols):
            if grid[r,c]==0 and not shines[r,c]: return False
    for r in range(rows):
        for c in range(cols):
            if lights[r,c]==2:
                for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    rr,cc=r+dr,c+dc
                    while 0<=rr<rows and 0<=cc<cols:
                        if grid[rr,cc]==-1 or grid[rr,cc]==10: break
                        if lights[rr,cc]==2: return False
                        rr+=dr; cc+=dc
    remain = remaining_number(lights)
    if any(v>0 for v in remain.values()): return False
    return True

# --- 状態ベクトル ---
def get_state(lights): return tuple(lights.flatten())

# --- Q学習報酬 ---
def get_reward(lights, action):
    temp = lights.copy()
    temp[action]=2
    reward = np.sum(compute_shine(temp))
    remain = remaining_number(temp)
    for v in remain.values(): 
        if v<0: reward-=5
    if check_clear(temp): reward+=100
    return reward

# --- Q学習 ---
Q = {}
alpha=0.1; gamma=0.9; epsilon=0.3; episodes=3200

for ep in range(episodes:=800):
    lights = np.zeros_like(grid)
    for step in range(25):
        state = get_state(lights)
        if state not in Q: Q[state]={}
        shines = compute_shine(lights)
        actions = [(r,c) for r in range(rows) for c in range(cols)
                   if grid[r,c]==0 and lights[r,c]==0 and not shines[r,c]
                   and all(remaining_number(lights).get((r+dr,c+dc),0)>0
                           for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]
                           if 0<=r+dr<rows and 0<=c+dc<cols and grid[r+dr,c+dc]==10)]
        if not actions: break
        if random.random()<epsilon:
            action=random.choice(actions)
        else:
            q_vals={a:Q[state].get(a,0) for a in actions}
            action=max(q_vals,key=lambda x:q_vals[x])
        reward = get_reward(lights, action)
        lights[action]=2
        next_state=get_state(lights)
        Q[state][action]=Q[state].get(action,0)+alpha*(reward+gamma*max(Q.get(next_state, {}).values(), default=0)-Q[state].get(action,0))
        if reward>=100: break

# --- 学習結果で解を作る ---
lights = np.zeros_like(grid)
history=[lights.copy()]
for _ in range(25):
    state = get_state(lights)
    shines = compute_shine(lights)
    actions = [(r,c) for r in range(rows) for c in range(cols)
               if grid[r,c]==0 and lights[r,c]==0 and not shines[r,c]
               and all(remaining_number(lights).get((r+dr,c+dc),0)>0
                       for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]
                       if 0<=r+dr<rows and 0<=c+dc<cols and grid[r+dr,c+dc]==10)]
    if not actions: break
    q_vals={a:Q.get(state,{}).get(a,0) for a in actions}
    action=max(q_vals,key=lambda x:q_vals[x])
    lights[action]=2
    history.append(lights.copy())
    if check_clear(lights): break

# === 描画関数 ===
def draw_board(ax, lights=None, shines=None, message=None):
    ax.clear()
    ax.set_xticks(np.arange(cols+1)-0.5, minor=True)
    ax.set_yticks(np.arange(rows+1)-0.5, minor=True)
    ax.grid(which="minor", color="k", linewidth=1)
    ax.set_xlim(-0.5, cols-0.5)
    ax.set_ylim(rows-0.5, -0.5)
    ax.set_aspect("equal")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    remain = remaining_number(lights) if lights is not None else {}
    for r in range(rows):
        for c in range(cols):
            cell=grid[r,c]
            if cell==-1: ax.add_patch(plt.Rectangle((c-0.5,r-0.5),1,1,color="black"))
            elif cell==10:
                ax.add_patch(plt.Rectangle((c-0.5,r-0.5),1,1,color="black"))
                val=remain.get((r,c),original_numbers.get((r,c),0))
                text="😄" if val==0 else str(val)
                ax.text(c,r,text,color="white",ha="center",va="center",fontsize=14,weight="bold")
            elif shines is not None and shines[r,c]: ax.add_patch(plt.Rectangle((c-0.5,r-0.5),1,1,color="yellow",alpha=0.3))
            if lights is not None and lights[r,c]==2: ax.plot(c,r,"o",color="orange",markersize=18)
    if message: ax.text(cols/2-0.5,rows+0.3,message,color="green",fontsize=18,ha="center")

# === アニメ表示 ===
fig, ax = plt.subplots()
rects=[]
for r in range(rows):
    row_rects=[]
    for c in range(cols):
        cell=grid[r,c]
        color="black" if cell==-1 or cell==10 else "white"
        rect=plt.Rectangle((c-0.5,r-0.5),1,1,color=color)
        ax.add_patch(rect)
        row_rects.append(rect)
    rects.append(row_rects)

dots=[]
texts=[]
def update(frame):
    lights = history[frame]
    shines = compute_shine(lights)
    remain = remaining_number(lights)
    # 光の表示
    for r in range(rows):
        for c in range(cols):
            if grid[r,c]==0: rects[r][c].set_facecolor("yellow" if shines[r,c] else "white")
    # 電球の表示
    for dot in dots: dot.remove()
    dots.clear()
    for r in range(rows):
        for c in range(cols):
            if lights[r,c]==2: dots.append(ax.plot(c,r,"o",color="orange",markersize=18)[0])
    # 数字マスの残り電球数更新
    for t in texts: t.remove()
    texts.clear()
    for (r,c),v in remain.items():
        text = "😄" if v==0 else str(v)
        texts.append(ax.text(c,r,text,color="white",ha="center",va="center",fontsize=14,weight="bold"))
    return rects + dots + texts

ani = animation.FuncAnimation(fig, update, frames=len(history), interval=500, repeat=False)
plt.show()

# --- 初期盤面でのQ値ヒートマップ ---
initial_state=get_state(np.zeros_like(grid))
action_q=Q.get(initial_state,{})
q_grid=np.full((rows,cols),np.nan)
for (r,c),val in action_q.items(): q_grid[r,c]=val

plt.figure(figsize=(5,5))
plt.imshow(q_grid,cmap="coolwarm")
for r in range(rows):
    for c in range(cols):
        if not np.isnan(q_grid[r,c]): plt.text(c,r,f"{q_grid[r,c]:.1f}",ha="center",va="center",color="black")
plt.colorbar(label="Q value")
plt.title("Q values for initial state")
plt.show()

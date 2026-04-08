import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'server')
from environment import NutrisyncEnv
from models import NutrisyncAction, IngredientItem

# === BALANCED PLAN ===
env = NutrisyncEnv(difficulty='medium', calorie_target=2000, protein_target=100, budget=1500)
env.reset()
meals = [
    [('oats',80),('banana',100),('milk',150)],
    [('chicken',150),('rice',150),('spinach',100)],
    [('paneer',120),('roti',100),('tomato',80)],
    [('curd',150),('guava',100)],
]
names = ['BRK','LUN','DIN','SNK']
scores = []
for n, items in zip(names, meals):
    a = NutrisyncAction(items=[IngredientItem(ingredient=i, quantity=q) for i,q in items])
    _, r, done, info = env.step(a)
    scores.append(r.score)
    print(n, round(r.score, 2))
s = env.get_episode_summary()
total_good = sum(scores)
print('BALANCED TOTAL:', round(total_good, 2))
print('Cal:', round(s['total_calories']), '/', s['target_calories'])
print('Prot:', round(s['total_protein']), '/', s['target_protein'])
print()

# === EXPLOIT: ghee spam ===
env2 = NutrisyncEnv(difficulty='medium', calorie_target=2000, protein_target=100, budget=1500)
env2.reset()
scores2 = []
for n in names:
    a = NutrisyncAction(items=[IngredientItem(ingredient='ghee', quantity=30)])
    _, r, done, _ = env2.step(a)
    scores2.append(r.score)
    print(n, 'GHEE:', round(r.score, 2))
total_bad = sum(scores2)
print('EXPLOIT TOTAL:', round(total_bad, 2))
print()
print('SEPARATION:', round(total_good - total_bad, 2), '(balanced - exploit)')
print('Anti-gaming working:', 'YES' if total_good > total_bad * 2 else 'NEEDS TUNING')

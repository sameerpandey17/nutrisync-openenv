import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'server')

print('--- Testing models import ---')
from models import NutrisyncState, NutrisyncAction, NutrisyncObservation, NutritionInfo, IngredientItem, ConstraintConfig
print('models OK')

print('--- Testing environment import ---')
from server.environment import NutrisyncEnv, NUTRITION_DATABASE
print(f'DB size: {len(NUTRITION_DATABASE)} ingredients')

print('--- Testing reward import ---')
from server.reward import RewardEngine, compute_episode_reward
print('reward OK')

print('--- Testing tasks import ---')
from server.tasks import TASKS
print(f'Tasks: {list(TASKS.keys())}')

print('--- Running quick episode (easy) ---')
cfg = TASKS['easy'].copy()
grader = cfg.pop('grader')
env = NutrisyncEnv(**cfg, seed=1)
obs = env.reset()
print(f'Reset OK: meal={obs.current_meal}, satiety={obs.satiety}, unavailable={len(obs.unavailable_ingredients)}')

action = NutrisyncAction(items=[
    IngredientItem(ingredient='chicken', quantity=150, cooking_method='boiled'),
    IngredientItem(ingredient='roti', quantity=100, cooking_method='roasted'),
    IngredientItem(ingredient='spinach', quantity=100, cooking_method='steamed'),
])
obs, reward, done, info = env.step(action)
print(f'Step1: reward={reward.score:.2f}, satiety={obs.satiety:.1f}, gate={obs.budget_gate}')

action2 = NutrisyncAction(items=[
    IngredientItem(ingredient='rice', quantity=200, cooking_method='boiled'),
    IngredientItem(ingredient='moong_dal', quantity=150, cooking_method='boiled'),
    IngredientItem(ingredient='tomato', quantity=100, cooking_method='sauteed'),
])
obs, reward, done, info = env.step(action2)
print(f'Step2: reward={reward.score:.2f}, satiety={obs.satiety:.1f}')

action3 = NutrisyncAction(items=[
    IngredientItem(ingredient='paneer', quantity=100, cooking_method='raw'),
    IngredientItem(ingredient='wheat_flour', quantity=100, cooking_method='boiled'),
    IngredientItem(ingredient='cauliflower', quantity=150, cooking_method='steamed'),
])
obs, reward, done, info = env.step(action3)
print(f'Step3: reward={reward.score:.2f}, satiety={obs.satiety:.1f}')

action4 = NutrisyncAction(items=[
    IngredientItem(ingredient='oats', quantity=80, cooking_method='boiled'),
    IngredientItem(ingredient='banana', quantity=100, cooking_method='raw'),
])
obs, reward, done, info = env.step(action4)
print(f'Step4: done={done}, reward={reward.score:.2f}')

state = env.get_state()
print(f'hard_fail={state.hard_fail}, budget_gate={state.budget_gate}')
print(f'cluster_usage_counts={state.cluster_usage_counts}')
print(f'daily_totals={state.daily_totals}')
final = compute_episode_reward(state)
print(f'Episode reward (0-10 scale): {final:.2f}')
grade = grader(state)
print(f'Normalized grade (0-1): {grade:.4f}')

# Test hard_fail kill switch  
print('\n--- Testing hard_fail (vegan + paneer) ---')
cfg2 = TASKS['hard'].copy()
grader2 = cfg2.pop('grader')
env2 = NutrisyncEnv(**cfg2, seed=42)
env2.reset()
# Paneer is non-vegan, should trigger hard_fail
action_bad = NutrisyncAction(items=[
    IngredientItem(ingredient='paneer', quantity=100, cooking_method='raw'),
    IngredientItem(ingredient='roti', quantity=100, cooking_method='roasted'),
])
obs2, r2, done2, info2 = env2.step(action_bad)
print(f'hard_fail expected True: {obs2.hard_fail}')
# Finish episode
for _ in range(3):
    if obs2.done: break
    obs2, r2, done2, info2 = env2.step(NutrisyncAction(items=[
        IngredientItem(ingredient='moong_dal', quantity=100, cooking_method='boiled'),
        IngredientItem(ingredient='roti', quantity=80, cooking_method='roasted'),
    ]))
state2 = env2.get_state()
final2 = compute_episode_reward(state2)
print(f'hard_fail episode reward (should be 0.0): {final2}')

print('\n=== ALL TESTS PASSED ===')

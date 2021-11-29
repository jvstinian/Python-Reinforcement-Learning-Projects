from agent import Agent
from helper import getStockData, getState, formatPrice
import math # TODO

window_size = 50
batch_size = 32
agent = Agent(window_size, batch_size)
data = getStockData("GSPC") # ("^GSPC")
# data = list(range(1,1001))

l = len(data) - 1
episode_count = 150
stop_on_drawdown = 0.3

for e in range(episode_count):
    # print("Episode " + str(e) + "/" + str(episode_count))
    state = getState(data, 0, window_size + 1)
    state[0][0] = (1-stop_on_drawdown) # TODO 
    state[0][1] = 0
    state[0][2] = 1/(1 + math.exp(data[0] - data[0]))
    
    agent.inventory = []
    total_profit = 0
    number_of_buys = 0
    number_of_sells = 0
    cash = 50000
    mv_min = (1 - stop_on_drawdown)*cash
    done = False
    for t in range(l):
        action = agent.act(state)
        action_prob = agent.actor_local.model.predict(state)

        next_state = getState(data, t + 1, window_size + 1)
        # print('Next state: %s' % (next_state))
        reward = 0

        if action == 1:
            if data[t] <= cash:
                agent.inventory.append(data[t])
                state[0][1] = len(agent.inventory)
                cash -= data[t]
                # print("Buy:" + formatPrice(data[t]))
                number_of_buys += 1

        elif action == 2 and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            state[0][1] = len(agent.inventory)
            cash += data[t]
            # reward = max(data[t] - bought_price, 0)
            total_profit += data[t] - bought_price
            # print("sell: " + formatPrice(data[t]) + "| profit: " + formatPrice(data[t] - bought_price))
            number_of_sells += 1

        reward = len(agent.inventory)*(data[t+1] - data[t])
        # print("Time %s, shares: %s, reward: %s" % (t, len(agent.inventory), reward))
        
        current_mv = cash + len(agent.inventory)*data[t]
        # if current_mv < mv_min:
        #     done = True
        #     for bought_price in agent.inventory:
        #         total_profit += data[t] - bought_price
        #         cash += data[t]
        #     agent.inventory.clear()
        #     state[0][1] = 0
        if t == l - 1:
            done = True
            for bought_price in agent.inventory:
                total_profit += data[t+1] - bought_price
                cash += data[t+1]
            agent.inventory.clear()
            state[0][1] = 0

        # mv_min = max((1-stop_on_drawdown)*current_mv, mv_min)
        next_state[0][0] = mv_min/current_mv
        next_state[0][1] = len(agent.inventory)
        next_state[0][2] = 1/(1 + math.exp(data[0] - data[t+1]))

        agent.step(action_prob, reward, next_state, done)
        state = next_state

        if done:
            print("------------------------------------------")
            print("Episode " + str(e) + "/" + str(episode_count))
            print("------------------------------------------")
            print("Total Profit: " + formatPrice(total_profit))
            print("Number of buys: %s" % (number_of_buys,))
            print("Number of sells: %s" % (number_of_sells,))
            print("Cash: %s" % (cash,))
            print("------------------------------------------", flush=True)
            if t != l - 1:
                break

test_data = getStockData("GSPC_test") # ("^GSPC Test")
# test_data = list(range(1001, 1252)) 
l_test = len(test_data) - 1
state = getState(test_data, 0, window_size + 1)
total_profit = 0
number_of_buys = 0
number_of_sells = 0
cash=50000
agent.inventory = []
agent.is_eval = False
done = False
for t in range(l_test):
    action = agent.act(state)

    next_state = getState(test_data, t + 1, window_size + 1)
    reward = 0

    if action == 1:
        if data[t] <= cash:
            agent.inventory.append(test_data[t])
            cash -= test_data[t]
            print("Buy: " + formatPrice(test_data[t]))
            number_of_buys += 1

    elif action == 2 and len(agent.inventory) > 0:
        bought_price = agent.inventory.pop(0)
        cash += test_data[t]
        # reward = max(test_data[t] - bought_price, 0)
        total_profit += test_data[t] - bought_price
        print("Sell: " + formatPrice(test_data[t]) + " | profit: " + formatPrice(test_data[t] - bought_price))
        number_of_sells += 1

    reward = len(agent.inventory)*(test_data[t+1] - test_data[t])

    if t == l_test - 1:
        done = True
        for bought_price in agent.inventory:
            cash += test_data[t+1]
            total_profit += test_data[t+1] - bought_price
        agent.inventory.clear()

    agent.step(action_prob, reward, next_state, done)
    state = next_state

    if done:
        print("------------------------------------------")
        print("Test Data")
        print("------------------------------------------")
        print("Total Profit: " + formatPrice(total_profit))
        print("------------------------------------------")
        print("Number of buys: %s" % (number_of_buys,))
        print("Number of sells: %s" % (number_of_sells,))
        print("Cash: %s" % (cash,))
        print("------------------------------------------")


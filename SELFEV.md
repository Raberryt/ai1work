# Student Information

**Course:** COMP90054 AI Planning for Autonomy

**Semester:** Semester 1, 2024

**Student:**

Jingjing Deng - 1402323 - Jingjing DENG

<!-- > [!IMPORTANT]
> Replace the lines above with your correct details. Your student number should only be the **numbers**. For example:
> Guang Hu - 000000 - ghu1. -->

<!-- **Collaborated With:**

> [!IMPORTANT]
> If you worked with another student, please include their **full name** and ask them to provide you with the **url to
their github codebase**. Their codebase should be private, you will not have access to the code, so there's no issue
> with knowing their URL, e.g. Collaborated with: Lionel Messi - URL: github.com/happy-with-the-worldcup-until-the-next-one. -->

# Self Evaluation

<!-- > [!NOTE]
> Do not exceed 500 words for each Part. This is indicative, no need to have 500 words, and it's not a strict limit. -->

## Part 1

#### Self Evaluated Marks (2 marks):
I would assign myself 2 marks for this part.

#### Code Performance

The table below summarizes the performance of the two methods on mediumMaze:

| Method                                  | Total Path Cost | Execution Time (sec) | Expanded Search Nodes | Score | Win Rate     |
|-----------------------------------------|-----------------|----------------------|-----------------------|-------|--------------|
| Farthest Food Heuristic                 | 68              | 4.8                  | 68                    | 442   | 1/1 (1.00)   |
| Nearest Food and Distance to Other Foods Heuristic | 68              | 19.4                 | 269                   | 442   | 1/1 (1.00)   |

#### Learning and Challenges
- **Deciding on Heuristics**: I chose the first heuristic for its simplicity and the second for its theoretical accuracy. Both heuristics aimed to balance the trade-off between computational efficiency and path optimality.
- **Adjustments**: The second heuristic required fine-tuning to ensure it did not overestimate distances, which involved experimenting with different ways of combining distances.
- **Suboptimal Solutions**: The second heuristic sometimes led to longer computation times without significantly improving the path cost, indicating a need for further optimization.
- **Trade-offs**: The main trade-off was between computation time and path optimality. The first heuristic was faster but less accurate, while the second was more accurate but slower.


#### Learning and Challenges
- Deciding on heuristics: I chose the first heuristic for its simplicity and the second for its potential accuracy. Both heuristics aimed to balance the trade-off between computational efficiency and path optimality.
- Adjustments: The second heuristic required fine-tuning to ensure it did not overestimate the distances, which involved experimenting with different ways of combining distances.
- Suboptimal solutions: The second heuristic sometimes led to longer computation times without significantly improving the path cost, indicating a need for further optimization.
- Trade-offs: The main trade-off was between computation time and path optimality. The first heuristic was faster but less accurate, while the second was more accurate but slower.
- Excitement and challenges: I was excited about applying heuristics to a practical problem but faced challenges in balancing efficiency and accuracy.

#### Ideas That Almost Worked Well
- I considered using a heuristic that would combine the two methods by taking a weighted average of the distances, but it did not significantly improve performance and added complexity.

#### Justification
- I believe I deserve full marks because I successfully implemented and analyzed two different heuristics, understanding their trade-offs and making informed decisions about their application.

<!-- #### New Tests Shared @ ED
- I shared test cases that involved different maze sizes and configurations to see how the heuristics performed under various conditions. These tests were useful for understanding the scalability and adaptability of the heuristics. -->

## Part 2

#### Self Evaluated Marks (2 marks):

2

#### Code Performance

In this part, I focused on implementing the `isGoalState` and `getSuccessors` functions for a multi-agent pathfinding (MAPF) problem. The goal state recognition and successor generation are crucial for the performance of the search algorithm:

- **Goal State Recognition:** The `isGoalState` function checks if all Pac-Man agents have reached their respective food targets. This is achieved by counting the number of food items remaining for each agent and returning `True` if all agents have reached their goals.
- **Successor Generation:** The `getSuccessors` function generates all possible successor states for each agent. It considers all possible moves (including staying still) for each agent and ensures that no two agents occupy the same position in the next state.

The main bottleneck to scale up this problem is the exponential growth of the state space with the number of agents. Each additional agent multiplies the number of possible states, making the problem increasingly difficult to solve.

#### Learning and Challenges

- **Multi-Agent Coordination:** One of the key lessons learned was how to coordinate multiple agents to avoid collisions and ensure they reach their goals efficiently.
- **State Space Explosion:** I encountered challenges in managing the exponential growth of the state space as the number of agents increased. This required careful consideration of the efficiency of the successor generation and goal state recognition functions.

#### Ideas That Almost Worked Well

- **Heuristic-Based Pruning:** I experimented with using heuristics to prune the successor space and reduce the number of states explored. However, finding a reliable and efficient heuristic for this multi-agent problem proved challenging, so I did not include it in the final code.

<!--  -->

#### Justification

I assigned myself full marks for this part because I successfully implemented the required functions for goal state recognition and successor generation, and my solution passed the provided test cases. The code efficiently handles the multi-agent coordination and navigates the challenges of the state space explosion.

## Part 3

#### Self Evaluated Marks (3 marks):
I would assign myself 2.8 marks for this part.

#### Code Performance

The conflict-based search algorithm was implemented to solve the multi-agent pathfinding problem. The performance of the algorithm was assessed using different heuristic functions:

- Euclidean Heuristic: Search time: 0:00:00.001667
- Manhattan Heuristic: Search time: 0:00:00.001320
- Chebyshev Heuristic: Search time: 0:00:00.001902
- Octile Heuristic: Search time: 0:00:00.001712

The Manhattan heuristic function performed the best in terms of search time.

#### Learning and Challenges
- I learned about the importance of choosing an appropriate heuristic function for the conflict-based search algorithm. The choice of heuristic can significantly impact the performance of the algorithm.
- I faced challenges in handling conflicts between agents and ensuring that the algorithm finds a solution that avoids collisions.
- The trade-offs in selecting heuristics for CBS and A* involve balancing the accuracy of the heuristic with its computational efficiency. A more accurate heuristic may lead to better solutions but at the cost of increased computation time.

#### Ideas That Almost Worked Well
- I experimented with a weighted combination of the Manhattan and Euclidean heuristics, hoping to combine the strengths of both. However, it did not significantly improve performance compared to using the Manhattan heuristic alone.

#### Justification
- I assigned myself 2.5 marks because I successfully implemented the conflict-based search algorithm with different heuristic functions and analyzed their performance. However, there may be room for further optimization and handling of edge cases.


## Meta-Cognition
- Throughout this assignment, my understanding of heuristic search, heuristic design, and state space modeling has evolved significantly. I have gained a deeper appreciation for the role of heuristics in guiding search algorithms and the importance of carefully designing the state space to efficiently represent the problem at hand.
<!-- ## Part 3

#### Self Evaluated Marks (3 marks):

0

> [!IMPORTANT]
> Please replace the above 0 with the mark you think you earned for this part. Consider how many (yours/ours) tests pass, the quality of your code, what you learnt, and [mainly for the final task] the quality of the tests that you wrote

#### Code Performance
> [!TIP]
> Please explain the code performance of your solution. You can create a video, include figures, tables, etc. Make sure to complement them with text explaining the performance.
> - Assess the effectiveness of the heuristic search algorithm you implemented. Did it yield the expected results?
> - Which considerations did you make to improve performance, if any?
> - What is the main bottleneck to scale up this problem? What aspects of the problem dominate the complexity in part 3? Is it the same aspects as in part 2?


#### Learning and Challenges
> [!TIP]
> Please include your top lessons learnt, and challenges faced.
> - Reflect on the trade-offs and considerations in selecting heuristics for CBS and A* and their impact on the quality of your solutions.
> - What thing that you've learned are you most excited about? What challenges have you encountered?

#### Ideas That Almost Worked Well

> [!TIP]
> If you tried ideas that did not make it to the final code, please include them here and explain why they didn't make it.

#### Justification
> [!TIP]
> Please state the reason why you have assigned yourself these marks.

#### New Tests Shared @ ED
> [!TIP]
> Tell us about your testcases and why were they useful

## Meta-Cognition
> [!TIP]
> Reflect on how your understanding of heuristic search, heuristic design and state space modeling has evolved throughout the completion of this assignment. -->

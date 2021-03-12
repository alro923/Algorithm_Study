# 최단경로
- 가장 짧은 경로를 찾는 알고리즘 (a.k.a 길 찾기 문제)
- 그래프를 이용해 표현할 수 있다. -> 그림 그려보기!
    - 노드 : 각 지점
    - 간선 : 각 지점 사이에 연결된 도로

## 1. 다익스트라 최단 경로 알고리즘
- **한** 지점에서 다른 **특정** 지점까지의 최단 경로를 구해야 하는 경우
- '음의 간선'이 없을 때 정상적으로 작동 (현실에서는 있을 수 없음)
- 매번 '가장 비용이 적은 노드'를 선택해서 어떤 과정을 반복하기 때문에, '그리디 알고리즘'으로 분류된다.

### 작동방식
```
1. 출발 노드를 설정한다.
2. 최단 거리 테이블을 초기화한다.
3. 방문하지 않은 노드 중에서 최단거리가 가장 짧은 노드를 선택한다.
4. 해당 노드를 거쳐 다른 노드로 가는 비용을 계산하여, 최단 거리 테이블을 갱신한다.
5. 위 과정에서 3, 4번을 반복한다.
```
- 다익스트라 알고리즘은 최단 경로를 구하는 과정에서 '각 노드에 대한 현재까지의 최단 거리' 정보를 항상 1차원 리스트에 저장하며 리스트를 계속 갱신한다.
- 매번 현재 처리하고 있는 노드를 기준으로 주변 간선을 확인해서, 더 짧은 노드가 있으면 그걸로 갱신한다.
- 따라서, 방문하지 않은 노드 중, 현재 최단 거리가 가장 짧은 노드를 찾아서,  4번을 수행하므로 '그리디 알고리즘' 으로 볼 수 있다.


### (1) 구현하기 쉽지만 느리게 동작하는 코드
- O(V^2)
- 처음에 각 노드에 대한 최단 거리를 담는 1차원 리스트 선언
- 단계마다 '방문하지 않은 노드 중에서 최단 거리가 가장 짧은 노드를 선택'하기 위해, 매 단계마다 1차원 리스트의 모든 원소를 확인(순차 탐색)한다.

#### 간단한 다익스트라 알고리즘 소스코드
```py
import heapq
import sys

INF = int(1e9)

n, m = map(int, input().split())
start = int(input())
graph = [[] for i in range(n+1)]

visited = [False]*(n+1)
distance = [INF] *(n+1)

for _ in range(m):
  a, b, c = map(int, input().split())
  graph[a].append((b,c))

def get_smallest_node():
  min_value = INF
  index = 0
  for i in range(1, n+1):
    if distance[i] < min_value and not visited[i]:
      min_value = distance[i]
      index = i
  return index

def dijkstra(start):
  distance[start] = 0
  visited[start] = True
  for j in graph[start]:
    distance[j[0]] = j[1]
  for i in range(n-1):
    now = get_smallest_node()
    visited[now] = True
    for j in graph[now]:
      cost = distance[now] + j[1]
      if cost < distance[j[0]]:
        distance[j[0]] = cost
                 
dijkstra(start)

for i in range(1, n+1):
  if distance[i] == INF:
    print('INFINITY')
  else:
    print(distance[i])
``` 
#### 💡 설명
- 시간 복잡도 : O(V^2)
  - V번에 걸쳐서 최단거리가 가장 짧은 노드를 매번 선형탐색, 현재 노드와 연결된 노드 (최대) V개 매번 확인
- 최단 경로 문제에서 전체 노드의 개수가 5000개 이하면 ㄱㅊ, 근데 10000개 넘어가면 시간 너무 걸려서 '개선된 다익스트라 알고리즘'을 이용해야 한다. 


### (2) 개선된 다익스트라 알고리즘
- 시간 복잡도 : O(ElogV)
- (1)에서는 최단거리를 선형적으로(모든 원소를 하나씩) 탐색했음. 그래서 이 과정에서 O(V) 시간이 걸림
- 그치만 최단 거리가 가장 짧은 노드를 더욱더 빠르게 찾을 수 있다면 알고리즘쨩 시간복잡도를 더 줄일 수 있지 않을까?
- 어떻게? 힙(Heap) 자료구조 사용!
  - 특정 노드까지의 최단거리에 대한 정보를 힙에 담아서 처리하므로, 출발노드로부터 가장 거리가 짧은 노드를 더욱 빠르게 찾을 수 있다. -> 로그 시간이 걸린다. -> 획기적으로 빨라진다.

#### 힙에 대해서 설명
- 우선순위 큐 구현할 때 사용하는 자료구조 중 하나
- 스택 : Last In First Out
- 큐 : First In First Out
- **우선순위 큐** : 우선순위가 가장 높은 데이터를 가장 먼저 삭제
  - heapq 라이브러리 사용
  - 우선순위 값을 표현할 때, 일반적으로 정수형 자료형의 변수가 사용된다.
  - ex. 물건정보 : (가치, 물건 이름)
  - 우선순위 큐 라이브러리에 넣은 데이터의 묶음은 첫번째 원소를 기준으로 우선순위를 설정
  - ex. 물건정보는 '가치'가 우선순위 값이 된다.
  - '최소 힙' : '값이 낮은 데이터가 먼저 삭제'
    - 파이썬 라이브러리에서는 기본적으로 최소 힙 구조 사용
    - 다익스트라 최단 경로 알고리즘에서는 비용이 적은 노드를 우선시하여 방문 -> 최소 힙 그대로 사용하면된다!
    - '최대 힙'으로 사용하려면, 일부러 우선순위에 해당하는 값에 (-) 를 붙여서 넣었다가, 나중에 우선순위 큐에서 꺼낸 다음에 다시 (-) 를 붙여서 원래의 값으로 되돌리면 된다.
  - cf. 우선순위 큐는 리스트를 이용해서 구현할 수도 있다.

  | 우선순위 큐 구현 방식 | 삽입 시간 | 삭제 시간|
  | ----- | --- | --- |
  |리스트 | O(1) | O(N) |
  | 힙 | O(logN) | O(logN) |

- 힙 : 전체 시간 복잡도 O(NlogN)
- 리스트 : 전체 시간 복잡도 O(N^2)

- 최단거리를 저장하기 위한 1차원 리스트 (최단 거리 테이블)는 (1)과 같이 그대로 이용하고, 현재 가장 가까운 노드를 저장하기 위한 목적으로만 우선순위 큐를 추가로 이용한다고 보면 된다.

- (1)에서 썼던 get_smallest_node()라는 함수를 작성할 필요 없었다. 왜냐면 '최단 거리가 가장 짧은 노드'를 선택하는 과정을 다익스트라 최단 경로 함수 안에서 우선순위 큐를 이요하는 방식으로 대체할 수 있기 때문이다!

#### 개선된 다익스트라 알고리즘 소스코드
```py
import heapq
import sys

INF = int(1e9)
# input = sys.stdin.readline
n, m = map(int, input().split())
start = int(input())
graph = [[] for i in range(n+1)]
distance = [INF]*(n+1)

for _ in range(m):
  a, b, c = map(int, input().split())
  graph[a].append((b,c))

def dijkstra(start):
  q = []
  heapq.heappush(q, (0, start))
  distance[start] = 0
  while q:
    dist, now = heapq.heappop(q)
    if distance[now] < dist:
      continue
    for i in graph[now] :
      cost = dist + i[1]
      if cost < distance[i[0]]:
        distance[i[0]] = cost
        heapq.heappush(q, (cost, i[0]))

dijkstra(start)

for i in range(1, n+1):
  if distance[i] == INF:
    print('INFINITY')
  else:
    print(distance[i])
```

## 2. 플로이드 워셜
- **모든** 지점에서 다른 **모든** 지점까지의 최단 경로를 모두 구해야 하는 경우
## 3. 벨만 포드 알고리즘

### 실전문제 3번 : 전보 (p.262)

#### 내 코드
```
n, m, c = map(int, input().split()) # 도시 갯수 n, 통로 갯수 m, 전보 보내는 도시 c
INF = int(1e9)
cnt = 0 # 도시 c에서 전보 보낼 수 있는 도시의 총 갯수
max_time = 0 # 최대 시간

graph = [[INF]*(n+1) for _ in range(n+1)]

for a in range(1, n+1):
  for b in range(1, n+1):
    if a == b:
      graph[a][b] = 0

for _ in range(m):
  x, y, z = map(int, input().split())
  # x 에서 y 까지 걸리는 시간 z
  graph[x][y] = z

for k in range(1, n+1):
  for a in range(1, n+1):
    for b in range(1, n+1):
      graph[a][b] = min(graph[a][b], graph[a][k] + graph[k][b])

for b in range(1, n+1):
  if graph[c][b] < INF and b != c:
    cnt = cnt +1
    max_time = max(max_time, graph[c][b])

print(cnt, max_time)

```

#### 결과
```
>>> 3 2 1
>>> 1 2 4
>>> 1 3 2
2 4
```

#### 💡 설명
- 플로이드-워셜로 풀었음
- INF 아니고 0 (제자리) 아니면 연결된 거니까, cnt 증가시켜줌
- graph 테이블 중에 들어있는 값들 중 가장 큰 값 (INF 제외)이 max_time

#### 🥅 개선할 점
- 도시 C에서 출발하는 것만 보면 되니까, **한** 도시에서 다른 도시까지의 최단 거리 문제로 치환할 수 있다. 
- 따라서 플로이드-워셜이 아닌 다익스트라 알고리즘으로 풀 수도 있다. (나는 플로이드-워셜 배운 직후라 써먹고 싶어서 이걸로 풀었다!)
- N과 M의 범위가 충분히 크기 때문에, 다익스트라로 풀려면 우선순위 큐를 사용해야한다.
- input 받을 때도 readline 으로 받는 게 실행시간 측면에서 더 유리하다. (이건 그냥 해도 딱히 문제 없어서 계속 안쓰게 되는듯...! 개선할 때는 꼭 readline으로 받자)

⚡️ 개선된 코드
```

```
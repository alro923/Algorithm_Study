# 그래프 이론

## 10장에서 배울 알고리즘
- 크루스칼 알고리즘 (Kruskal Algorithms)
  - 그리디 알고리즘으로 분류된다.
- 위상 정렬 알고리즘 (Topology Algorithms)
  - 큐 / 스택을 알아야 구현할 수 있다.

### 숙지할 것
- '서로 다른 개체가 연결되어 있다.' -> 그래프 알고리즘을 떠올려야한다.
  - ex) 여러 개의 도시가 연결되어 있다.
- 트리 자료구조
  - 다익스트라 알고리즘에서, 우선순위 큐를 사용하기 위해 최소 힙 / 최대 힙을 이용했음
  - 최소 힙 -> 부모 노드가 자식 노드보다 크기가 작다 -> 트리 자료구조


### 그래프 vs 트리 (자료 구조)
🖤|그래프 | 트리
------------ |------------ | -------------
방향성 |방향 그래프 or 무방향 그래프 | **방향 그래프**
순환성 | 순환 및 비순환 | 비순환
루트 노드 존재 여부 | 루트 노드가 없음 | 루트 노드가 존재
노드간 관계성 | 부모와 자식 관계 없음 | 부모와 자식 관계
모델의 종류 | 네트워크 모델 | 계층 모델

</br>

### 인접 행렬 vs 인접 리스트 (그래프 구현 방법)
- 1. 인접 행렬 (Adjacency Matrix) : 2차원 배열
- 2. 인접 리스트 (Adjacency List) : 리스트

🖤|인접 행렬 | 인접 리스트
------------ |------------ | -------------
메모리 | O(V^2) | O(E)
시간* | O(1) | O(V)
- 노드 개수 V, 간선 개수 E인 그래프
- *특정 노드 A에서 다른 노드 B로 이어진 간선의 비용을 찾는데 소요되는 시간

#### 다익스트라 알고리즘 : 인접 리스트
  - 노드의 개수가 V개 일때, V개의 리스트를 만들어서 각 노드와 연결된 모든 간선에 대한 정보를 리스트에 저장했었음.
#### 플로이드 워셜 알고리즘 : 인접 행렬
  - 모든 노드에 대하여 다른 노드로 가는 최소 비용을 V^2 크기의 2차원 리스트에 저장한 뒤, 해당 비용을 갱신해서 최단 거리를 계산함.
#### 어떤 문제를 만나든, 메모리와 시간을 고려해서 알고리즘을 선택하자!!
  - ex) 최단 경로를 찾아야하는 문제
    - 노드의 개수가 적은 경우 : 플로이드 워셜
    - 노드와 간선의 개수가 모두 많은 경우 : 우선순위 큐 사용하는 다익스트라 알고리즘

## 서로소 집합 자료구조 (Disjoint Sets)
- 서로소 집합 == 공통 원소가 없는 두 집합
  - ex. {1,2} 와 {3,4} 는 서로소 집합
  - ex. {1,2} 와 {1,2} 는 서로소 집합이 아님
- 두 집합이 서로소 관계인지 확인한다 == 어떤 원소를 공통으로 가지고 있는지 확인한다.
- union 연산 : 2개의 집합을 하나로 합치는 연산
- find 연산 : 특정 원소가 어떤 집합에 속해 있는지 알려주는 연산
- 이 두 연산을 써서 union-find 자료구조라고 불리기도 한다.

### 서로소 집합 알고리즘

#### 기본적인 서로소 집합 알고리즘 소스코드 (개선 가능)
```py
# 부모테이블 parent 와 원소 x를 이용해서, x가 속한 집합의 루트 노드 리턴
# 이때, 한 집합은 하나의 루트 노드를 가지므로 같은 값을 리턴하면 같은 집합에 있는 것이다.
def find_parent(parent, x):
  if parent[x] != x: # 루트노드가 아닌경우
    return find_parent(parent, parent[x])
  return x

# 원소 a, b 가 들어있는 집합을 합친다.
def union_parent(parent, a, b):
  a = find_parent(parent, a)
  b = find_parent(parent, b)
  if a < b: 
    parent[b] = a # a가 b보다 작으면 a가 b의 부모
  else:
    parent[a] = b # b가 a보다 작으면 b가 a의 부모


v, e = map(int, input().split()) # 노드 갯수 v, 간선 갯수 e
parent = [0]*(v+1) # 부모 테이블 초기화

for i in range(1, v+1):
  parent[i] = i # 초기 부모 테이블 : 자기 자신을 부모 노드로!

for i in range(e):
  a, b = map(int, input().split())
  union_parent(parent, a,b)

# 각 원소가 속한 집합 출력
for i in range(1, v+1):
  print('원소', i, '는 루트노드가 ', find_parent(parent, i), '인 집합에 속함')

# 부모 테이블 내용 출력
for i in range(1, v+1):
  print(parent[i], end=' ')

```

#### 결과
```
>> 6 4
>> 1 4
>> 2 3
>> 2 4
>> 5 6
원소 1 는 루트노드가  1 인 집합에 속함
원소 2 는 루트노드가  1 인 집합에 속함
원소 3 는 루트노드가  1 인 집합에 속함
원소 4 는 루트노드가  1 인 집합에 속함
원소 5 는 루트노드가  5 인 집합에 속함
원소 6 는 루트노드가  5 인 집합에 속함
1 1 2 1 5 5 
```

#### 💡 설명
- 전체 원소가 {1, 2, 3, 4} 와 {5, 6} 으로 나누어지는 것을 알 수 있다.

#### 🥅 개선할 점
- find 함수가 비효율적으로 동작해서, 최악의 경우 find 함수가 모든 노드를 다 확인하느라 시간복잡도가 O(V) 가 된다.
  - ex. (4,5), (3,4), (2,3), (1,2) 이면 1<-2<-3<-4<-5 로 된다.
- 결과적으로 연산 개수가 M이면, 전체 시간복잡도는 O(VM)가 된다.
- '경로 압축 기법 소스코드'를 사용해보자.
```py
def find_parent(parent, x):
  if parent[x] != x:
    parent[x] = find_parent, parent[x]
  return parent[x] # 원래 여기가 x 였음
``` 
- 이러면 find 함수 호출 후에 해당 노드의 루트 노드가 바로 부모 노드로 된다.
  - ex. (4,5), (3,4), (2,3), (1,2) 이면 1<-2, 1<-3, 1<-4, 1<-5 이렇게!
  - 쨌든 루트 다 같으니까 {1,2,3,4,5} 인건 변함 X

#### ⚡️ 경로 압축 기법으로 개선된 서로소 집합 알고리즘 소스코드
```py
def find_parent(parent, x):
  if parent[x] != x: # 루트노드가 아닌경우
    return find_parent(parent, parent[x])
  return parent[x] # 여기만 바꿨음

# 원소 a, b 가 들어있는 집합을 합친다.
def union_parent(parent, a, b):
  a = find_parent(parent, a)
  b = find_parent(parent, b)
  if a < b: 
    parent[b] = a # a가 b보다 작으면 a가 b의 부모
  else:
    parent[a] = b # b가 a보다 작으면 b가 a의 부모


v, e = map(int, input().split()) # 노드 갯수 v, 간선 갯수 e
parent = [0]*(v+1) # 부모 테이블 초기화

for i in range(1, v+1):
  parent[i] = i # 초기 부모 테이블 : 자기 자신을 부모 노드로!

for i in range(e):
  a, b = map(int, input().split())
  union_parent(parent, a,b)

# 각 원소가 속한 집합 출력
for i in range(1, v+1):
  print('원소', i, '는 루트노드가 ', find_parent(parent, i), '인 집합에 속함')

# 부모 테이블 내용 출력
for i in range(1, v+1):
  print(parent[i], end=' ')

```


#### 서로소 집합 알고리즘을 활용한 사이클 판별 소스코드
```py
def find_parent(parent, x):
  if parent[x] != x:
    return find_parent(parent, parent[x])
  return parent[x] # 경로 압축

def union_parent(parent, a, b):
  a = find_parent(parent, a)
  b = find_parent(parent, b)
  if a < b: 
    parent[b] = a
  else:
    parent[a] = b

v, e = map(int, input().split())
parent = [0]*(v+1)

for i in range(1, v+1):
  parent[i] = i

# 여기부터 다름
cycle = False # 사이클 발생 여부 초기화
for i in range(e):
  a, b = map(int, input().split())
  if find_parent(parent, a) == find_parent(parent, b):
    cycle = True
    break # break 는 가장 가까운 반복문 탈주
  else:
    union_parent(parent, a,b)

if cycle:
  print('사이클 발생함')
else:
  print('사이클 발생 안함')
```

#### 결과
```
>> 3 3
>> 1 2
>> 1 3
>> 2 3
사이클 발생함
```

## 신장 트리 자료구조 (Spanning Tree)
  1. 모든 노드를 포함하면서
  2. 사이클이 존재하지 않는 그래프
- 이건 트리의 성립 조건과 같으므로, 신장 '트리'라고 부르는 것이다!

## 최소 신장트리 알고리즘
N개의 도시가 있을 때, 최소 비용으로 도로를 건설해서 특정 도시가 서로 연결되게 해야하는 문제
- ex. 3개의 도시 A,B,C가 있고, A와 B가 연결되게 해야한다. (A-C-B 도 A-B 연결된거임)
  - A와 B 사이 연결 : 23
  - B와 C 사이 연결 : 13
  - A와 C 사이 연결 : 25 이면,
  - 23 + 13, 23 + 25, 13 + 25 중 값이 최소가 되도록 하기 위해 A-B, B-C를 연결하면 된다.
### 크루스칼 알고리즘
- 대표적인 최소 신장 트리 알고리즘
- 모든 간선에 대해 정렬을 수행한 뒤에, 가장 거리가 짧은 간선부터 집합에 포함시긴다
  - = 현재 상황에서 지금 당장 좋은 것부터 선택 = 그리디 알고리즘
1. 간선 데이터를 비용에 따라 오름차순으로 정렬
2. 간선 하나씩 확인하면서 현재의 간선이 사이클 발생시키는지 확인
  - 2-1. 사이클 발생 X : 최소 신장 트리에 포함시킴
  - 2-2. 사이클 발생 O : 최소 신장 트리에 포함시키지 않음
3. 모든 간선에 대해 2. 반복

#### 크루스칼 알고리즘 소스코드
```py
def find_parent(parent, x):
  if parent[x] != x:
    return find_parent(parent, parent[x])
  return parent[x]

def union_parent(parent, a, b):
  a = find_parent(parent, a)
  b = find_parent(parent, b)
  if a < b: 
    parent[b] = a
  else:
    parent[a] = b

v, e = map(int, input().split())
parent = [0]*(v+1)

for i in range(1, v+1):
  parent[i] = i

# 여기부터 다름

edges = []
result = 0

for _ in range(e):
  a, b, cost = map(int, input().split())
  # 비용순으로 정렬하기 위해, 튜플의 첫번째 원소 (cost) 를 비용으로 설정
  edges.append((cost, a, b))

edges.sort() #cost 순으로 정렬됨

for edge in edges :
  cost, a, b = edge
  if find_parent(parent, a ) != find_parent(parent, b):
    union_parent(parent, a, b)
    result += cost

print(result)
```

#### 결과
```
>> 7 9
>> 1 2 29
>> 1 5 75
>> 2 3 35
>> 2 6 34
>> 3 4 7
>> 4 6 23
>> 4 7 13
>> 5 6 53
>> 6 7 25
159
```

#### 💡 설명
- 간선 개수가 E개일 때, 시간복잡도 O(ElogE)
- 간선을 정렬하는 작업이 가장 오래걸리는데 E개면 ElogE 됨
- 크루스칼 내부에서 사용되는 알고리즘 시간 : 정렬 알고리즘 > 서로소 집합 알고리즘 (무시!)


## 위상 정렬 (Topology Sort)
- 정렬 알고리즘
- 방향 그래프의 모든 노드를 '방향성에 거스르지 않도록' 순서대로 나열하는 것
- ex. 컴공 커리에서 선수과목 있는거 (C언어 -> 자료구조-> 알고리즘)

> ### 진입 차수 (Indegree)
> - 특정한 노드로 '들어오는' 간선의 개수
> - ex. 선수과목 개수라고 생각하면됨. 자료구조는 진입차수 1, 알고리즘은 진입차수 2

- 위상정렬 알고리즘 작동 방식
1. 진입차수가 0인 노드를 큐에 넣는다.
2. 큐가 빌 때까지 다음의 과정을 반복한다.
  2-1. 큐에서 원소를 꺼내, 해당 노드에서 출발하는 간선을 그래프에서 제거한다.
  2-2. 새롭게 진입차수가 0이 된 노드를 큐에 넣는다.

- 사이클에 포함되어 있는 원소는 큐에 못 들어간다.
- 그니까 모든 원소를 방문하기 전에 (= 큐에서 원소 V번 추출되기 전에) 큐가 비면 사이클이 있다는 뜻이다.
- 문제 전제조건에서 '위상 정렬 문제에는 사이클이 발생하지 않는다'라고 명시하기도 한다.


#### 소스코드
```py
from collections import deque

v, e = map(int, input().split()) # 노드 개수 v, 간선 개수 e
indegree = [0] *(v+1) # 모든 노드에 대한 진입차수 0으로 초기화
graph = [[] for i in range(v+1)] # 각 노드에 연결된 간선정보를 담기위한 연결리스트(그래프) 초기화

for _ in range(e):
  a, b = map(int, input().split()) 
  graph[a].append(b) # a->b 인 간선정보 저장
  indegree[b] +=1 # b 진입차수 1 증가

# 위상 정렬 함수
def topology_sort():
  result = [] # 알고리즘 수행 결과
  q = deque() # 큐 기능을 위해 deque 라이브러리 사용

  for i in range(1, v+1):
    if indegree[i] == 0:
      q.append(i) # 진입차수가 0인 노드들 큐에 삽입

  while q: # 큐가 빌 때까지
    now = q.popleft()
    result.append(now)
    for i in graph[now]: # 이러면 i는 now->i 인 노드 말하는거지
      indegree[i] -= 1 # 진입차수 1 빼주기
      if indegree[i] == 0:
        q.append(i) # 새롭게 진입차수 0 되는 노드들은 큐에 넣어주기
  
  for i in result:
    print(i, end=' ') # 결과 출력

topology_sort()
```

#### 결과
```
>> 7 8
>> 1 2
>> 1 5
>> 2 3
>> 2 6
>> 3 4
>> 4 7
>> 5 6
>> 6 4
1 2 5 3 6 4 7 
```

#### 💡 설명
- 위상 정렬 과정을 수행하는 동안 큐에서 빠져나간 노드들 순서 == 위상 정렬 수행한 결과
- 위상 정렬의 답안은 여러가지가 될 수 있다.
  - 만약 한 단계에서 새롭게 큐에 새롭게 들어가는 원소가 2개 이상이면 어떤걸 먼저 큐에 넣는지에 따라 답은 달라지지만, 어쨌든 옳은 결과이기 때문이다.
  - ex. 2 5 를 같이 넣으면, 1 2 5 3 6 4 7 도 답이고, 1 5 2 3 6 4 7 도 답이다.
- 위상정렬 시간복잡도 : O(V+E)
  - 차례대로 모든 노드를 확인하고, 해당 노드에서 출발하는 간선을 차례대로 제거해야한다.
  - 따라서 노드와 간선을 모두 확인하므로, O(V+E)의 시간이 소요된다.
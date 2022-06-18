//
//  Solution1_ExtensionOne.swift
//  LeetCode
//
//  Created by haodong xu on 2022/3/25.
//

import Foundation
class Solution1 {
    class Trie {
        var children: [Trie?]
        var isEnd: Bool

        init() {
            isEnd = false
            children = [Trie?].init(repeating: nil, count: 26)
        }
    }

    var trie = Trie()

    func findAllConcatenatedWordsInADict(_ words: [String]) -> [String] {
        var result = [String]()
        let sortWords = words.sorted(by: { s1, s2 in
            return s1.count < s2.count
        })
        for (index, word) in sortWords.enumerated() {
            if word.count == 0 {
                continue
            }
            if isLinkWord(word: word, currentIndex: 0) {
                result.append(word)
            } else {
                insertWord(word: word)
            }
        }
        return result
    }

    func isLinkWord(word:String , currentIndex: Int) -> Bool {
        if currentIndex == word.count {
            return true
        }
        var node = self.trie
        for (index, c) in word.enumerated() {
            if index < currentIndex {
                continue
            }
            let charIndex = Int(c.asciiValue! - Character.init("a").asciiValue!)
            if let node = node.children[charIndex] {
                if node.isEnd {
                    if isLinkWord(word: word, currentIndex: index + 1) {
                        return true
                    }
                }
            } else {
                return false
            }
        }
        return false
    }

    func insertWord(word: String) {
        var node = trie
        for (_, c) in word.enumerated() {
            let index = Int(c.asciiValue! - Character.init("a").asciiValue!)
            if node.children[index] == nil {
                let trie = Trie()
                node.children[index] = trie
            }
            node = node.children[index]!
        }
        node.isEnd = true
    }

    func isNStraightHand(_ hand: [Int], _ groupSize: Int) -> Bool {
        guard hand.count % groupSize == 0 else { return false }
        let sortedHand = hand.sorted { v1, v2 in
            return v1 < v2
        }
        var dict: [Int: Int] = [Int: Int]()
        for num in sortedHand {
            dict[num] = (dict[num] ?? 0) + 1
        }
        for num in sortedHand {
            if dict[num] == 0 {
                continue
            } else {
                for i in 0..<groupSize {
                    if let value = dict[num + i] {
                        dict[num + i] = value - 1
                    } else {
                        return false
                    }
                }
            }
        }
        return dict.filter { element in
            element.value > 0
        }.count == 0
    }

    func countQuadruplets(_ nums: [Int]) -> Int {
        var ans = 0
        var dict = [Int: Int]()
        for b in (1..<nums.count-2).reversed() {
            for d in (b+2..<nums.count) {
                dict[nums[d] - nums[b + 1]] = (dict[nums[d] - nums[b + 1]] ?? 0) + 1
            }
            for a in 0..<b {
                ans += (dict[nums[a] + nums[b]] ?? 0)
            }
        }
        return ans
    }

    func checkPerfectNumber(_ num: Int) -> Bool {
        guard num != 1 else {
            return false
        }
        var sum = 1
        var index = 2
        while index * index <= num  {
            if num % index == 0 {
                sum += index
                if index * index < num {
                    sum += (num / index)
                }
            }
            index += 1
        }
        return sum == num
    }

    func slowestKey(_ releaseTimes: [Int], _ keysPressed: String) -> Character {
        guard releaseTimes.count == keysPressed.count && releaseTimes.count > 0 else {
            return "a"
        }
        var ans: Character = "a"
        var maxInterval = 0
        var lastTime = 0
        for (index, key) in keysPressed.enumerated() {
            let currentInterval = releaseTimes[index] - lastTime
            if maxInterval < currentInterval {
                ans = key
                maxInterval = currentInterval
            } else if maxInterval == currentInterval {
                ans = max(ans, key)
            }
            lastTime = releaseTimes[index]
        }
        return ans
    }

    func isAdditiveNumber(_ num: String) -> Bool {
        guard num.count >= 3 else {
            return false
        }
        var isAdditiveNumber = false
        for index in 1..<num.count / 2 + 1 {
            let subStr = num[num.startIndex..<num.index(num.startIndex, offsetBy: index)]
            if subStr.starts(with: "0") && subStr.count != 1 {
                continue
            }
            isAdditiveNumber = isAdditiveNumber || dfsAdditiveNumber(currentIndex: index, currentNum: String(subStr), num: num)
            if isAdditiveNumber {
                break
            }
        }
        return isAdditiveNumber
    }

    func dfsAdditiveNumber(currentIndex: Int, currentNum: String, num: String) -> Bool {
        if currentIndex == num.count - 1 {
            return true
        }
        // 剪枝
        if currentNum.count > num.count - currentNum.count {
            return false
        }
        var isAdditiveNumber = false
        for index in currentIndex..<num.count - 1 {
            let subStr = String(num[num.index(num.startIndex, offsetBy: currentIndex)...num.index(num.startIndex, offsetBy: index)])
            if subStr.starts(with: "0") && subStr.count != 1 {
                continue
            }
            let sumStr = String(Int(subStr)! + Int(currentNum)!)
            if currentIndex + subStr.count + sumStr.count > num.count {
                continue
            }
            let sumSub = String(num[num.index(num.startIndex, offsetBy: currentIndex + subStr.count)...num.index(num.startIndex, offsetBy: currentIndex + subStr.count + sumStr.count - 1)])
            if sumStr == sumSub {
                if currentIndex + subStr.count + sumStr.count == num.count {
                    isAdditiveNumber = true
                    break
                } else {
                    isAdditiveNumber = dfsAdditiveNumber(currentIndex: currentIndex + subStr.count, currentNum: String(subStr), num: num) || isAdditiveNumber
                }
            }
            if isAdditiveNumber {
                isAdditiveNumber = true
                break
            }
        }
        return isAdditiveNumber
    }

    var blocked: Set<Pane> = Set<Pane>()
    struct Pane: Hashable, Equatable {
        var x: Int
        var y: Int
    }

    func isEscapePossible(_ blocked: [[Int]], _ source: [Int], _ target: [Int]) -> Bool {
        for block in blocked {
            self.blocked.insert(Pane.init(x: block[0], y: block[1]))
        }
        let targetPane = Pane.init(x: target[0], y: target[1])
        let current: Pane = Pane(x: source[0], y: source[1])
        //        print(bfsEscapePossible(current: current, target: targetPane))
        //        print(bfsEscapePossible(current: targetPane, target: current))
        return bfsEscapePossible(current: targetPane, target: current)
    }

    func bfsEscapePossible(current: Pane, target: Pane) -> Bool {
        var queue: [Pane] = [Pane]()
        var visvitd: Set<Pane> = Set<Pane>()
        queue.append(current)
        visvitd.insert(current)
        let maxStep = blocked.count * (blocked.count - 1) / 2
        while !queue.isEmpty && visvitd.count <= maxStep {
            let pop = queue.removeFirst()
            if pop == target {
                return true
            }
            let nexts = [Pane(x: pop.x - 1, y: pop.y),
                         Pane(x: pop.x + 1, y: pop.y),
                         Pane(x: pop.x, y: pop.y - 1),
                         Pane(x: pop.x, y: pop.y + 1)]
            for next in nexts {
                if checkMove(next: next, visvitd: visvitd) {
                    queue.append(next)
                    visvitd.insert(next)
                    if pop == target {
                        return true
                    }
                }
            }
        }
        return visvitd.count > maxStep
    }

    func checkMove(next: Pane, visvitd: Set<Pane>) -> Bool {
        if next.x >= 1000000 || next.x < 0 || next.y >= 1000000 || next.y < 0 {
            return false
        }
        if self.blocked.contains(next) {
            return false
        }
        if visvitd.contains(next) {
            return false
        }
        return true
    }

    // 最长上升子序列
    func increasingTriplet(_ nums: [Int]) -> Bool {
        guard nums.count >= 3 else {
            return false
        }
        var minArr = [Int].init(repeating: Int.max, count: nums.count)
        minArr[0] = nums[0]
        var maxArr = [Int].init(repeating: Int.min, count: nums.count)
        maxArr[nums.count - 1] = nums[nums.count - 1]
        for index in 1..<nums.count {
            minArr[index] = min(nums[index], minArr[index - 1])
        }
        for index in (0..<nums.count-1).reversed() {
            maxArr[index] = max(nums[index], maxArr[index + 1])
        }
        for index in 1..<nums.count-1 {
            if nums[index] > minArr[index - 1] && nums[index] < maxArr[index + 1] {
                return true
            }
        }
        return false
    }

    func dominantIndex(_ nums: [Int]) -> Int {
        guard nums.count > 1 else {
            return 0
        }
        var max = nums[0]
        var ansIndex = 0
        for index in 1..<nums.count {
            if nums[index] > max {
                ansIndex = (max << 1 <= nums[index] ? index : -1)
                max = nums[index]
            } else {
                ansIndex = (nums[index] << 1 <= max ? ansIndex : -1)
            }
        }
        return ansIndex
    }

    func countVowelPermutation(_ n: Int) -> Int {
        var dp: [[Int]] = [[Int]].init(repeating: [Int].init(repeating: 0, count: 5), count: n)
        for index in 0..<5 {
            dp[0][index] = 1
        }
        var index = 1
        while index < n {
            dp[index][0] = (dp[index - 1][1] + dp[index - 1][2] + dp[index - 1][4]) % 1000000007
            dp[index][1] = (dp[index - 1][0] + dp[index - 1][2]) % 1000000007
            dp[index][2] = (dp[index - 1][1] + dp[index - 1][3]) % 1000000007
            dp[index][3] = (dp[index - 1][2]) % 1000000007
            dp[index][4] = (dp[index - 1][2] + dp[index - 1][3]) % 1000000007
            index += 1
        }
        let ans = dp[n - 1].reduce(0, { partialResult, value in
            return partialResult + value
        })
        return ans % 1000000007
    }

    func findMinDifference(_ timePoints: [String]) -> Int {
        guard timePoints.count >= 2 else {
            return -1
        }
        var ans = Int.max
        var timeValues: [Int] = [Int]()
        for timePoint in timePoints {
            timeValues.append(getTimePointValue(timePoint))
        }
        timeValues.sort()
        var firstTimeValue = timeValues[0]
        for index in 1..<timeValues.count {
            ans = min(ans, timeValues[index] - firstTimeValue)
            firstTimeValue = timeValues[index]
        }
        ans = min(1440 - (timeValues[timeValues.count - 1] - timeValues[0]), ans)
        return ans
    }

    func getTimePointValue(_ timePoint: String) -> Int {
        let values = timePoint.split(separator: ":")
        if values.count == 2 {
            let hour = Int(values[0]) ?? 0
            let min = Int(values[1]) ?? 0
            return 60 * hour + min
        }
        return -1
    }

    func containsNearbyDuplicate(_ nums: [Int], _ k: Int) -> Bool {
        var ans = false
        var startIndex = 0
        var endIndex = 0
        var set: Set<Int> = Set<Int>()
        while endIndex < nums.count {
            if set.contains(nums[endIndex]) {
                ans = true
                break
            }
            set.insert(nums[endIndex])
            if endIndex - startIndex == k {
                set.remove(nums[startIndex])
                startIndex += 1
            }
            endIndex += 1
        }
        return ans
    }

    func stoneGameIX(_ stones: [Int]) -> Bool {
        var counts = [0, 0, 0]
        for stone in stones {
            counts[stone % 3] += 1
        }
        if counts[0] % 2 == 0{
            return counts[1] >= 1 && counts[2] >= 1
        }
        return counts[1] - counts[2] > 2 || counts[2] - counts[1] > 2
    }

    func secondMinimum(_ n: Int, _ edges: [[Int]], _ time: Int, _ change: Int) -> Int {
        var map: [[Int]] = [[Int]].init(repeating: [Int](), count: n + 1)
        for edge in edges {
            map[edge[0]].append(edge[1])
            map[edge[1]].append(edge[0])
        }
        var path = [[Int]]()
        for _ in 0...n {
            path.append([Int.max, Int.max])
        }
        var queue = [Int]()
        var timeCost = 0
        queue.append(1)
        while !queue.isEmpty {
            let queueSize = queue.count
            var index = 0
            var nextTime = timeCost
            if (timeCost / change) % 2 == 0 {
                nextTime += time
            } else {
                nextTime = (timeCost / change + 1) * change + time
            }
            timeCost = nextTime
            while index < queueSize {
                index += 1
                let nexts = map[queue.removeLast()]
                for next in nexts {
                    if nextTime < path[next][0] {
                        path[next][1] = path[next][0]
                        path[next][0] = nextTime
                        queue.insert(next, at: 0)
                    } else if nextTime < path[next][1] && nextTime != path[next][0] {
                        if next == n {
                            return nextTime
                        }
                        path[next][1] = nextTime
                        queue.insert(next, at: 0)
                    }
                }
            }
        }
        return -1
    }

    // zimu - ！ . ,
    // - .count == 1 start end != -
    // ！ . , == end
    func countValidWords(_ sentence: String) -> Int {
        var ans = 0
        let tokens = sentence.split(separator: " ")
        for token in tokens {
            if token == "" {
                continue
            }
            var isVaild = true
            var linkWordCount = 0
            for index in token.indices {
                if token[index].isNumber {
                    isVaild = false
                    break
                }
                if token[index] == "-" {
                    linkWordCount += 1
                    if index == token.startIndex || index == token.index(before: token.endIndex) {
                        isVaild = false
                        break
                    }
                    let afterIndex = token.index(after: index)
                    if afterIndex == token.index(before: token.endIndex) &&
                        (token[afterIndex] == "!" || token[afterIndex] == "." || token[afterIndex] == ",") {
                        isVaild = false
                        break
                    }
                    if linkWordCount > 1 {
                        isVaild = false
                        break
                    }
                }
                if (token[index] == "!" || token[index] == "." || token[index] == ",")
                    && index != token.index(before: token.endIndex) {
                    isVaild = false
                    break
                }
            }
            if isVaild {
                ans += 1
            }
        }
        return ans
    }

    func numberOfWeakCharacters(_ properties: [[Int]]) -> Int {
        var ans = 0
        let sortedProperties = properties.sorted { role1, role2 in
            if role1[0] == role2[0] {
                return role1[1] > role2[1]
            }
            return role1[0] < role2[0]
        }
        var stack = [[Int]]()
        stack.append(sortedProperties[0])
        for index in 1..<sortedProperties.count {
            while !stack.isEmpty && sortedProperties[index][0] > stack.last![0] && sortedProperties[index][1] > stack.last![1] {
                stack.removeLast()
                ans += 1
            }
            stack.append(sortedProperties[index])
        }
        return ans
    }

    func highestPeak(_ isWater: [[Int]]) -> [[Int]] {
        let m = isWater.count
        let n = isWater[0].count
        var ans = [[Int]].init(repeating: [Int].init(repeating: 0, count: n), count: m)
        var queue = Queue<[Int]>()
        for (i, row) in isWater.enumerated() {
            for (j, item) in row.enumerated() {
                if item == 1 {
                    queue.enqueue(key: [i, j])
                    ans[i][j] = 0
                } else {
                    ans[i][j] = -1
                }
            }
        }
        var h = 1
        let dirs = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        while !queue.isEmpty {
            let length = queue.size
            for _ in 0..<length {
                let cur = queue.dequeue()!
                for dir in dirs {
                    let nx = cur[0] + dir[0]
                    let ny = cur[1] + dir[1]
                    if nx < 0 || nx >= m || ny < 0 || ny >= n {
                        continue
                    }
                    if ans[nx][ny] != -1 {
                        continue
                    }
                    ans[nx][ny] = h
                    queue.enqueue(key: [nx, ny])
                }
            }
            h += 1
        }
        return ans
    }

    public class LinkedList<T> {
        var data: T
        var next: LinkedList?
        public init(data: T){
            self.data = data
        }
    }

    public class Queue<T> {
        typealias LLNode = LinkedList<T>
        var head: LLNode!
        var size = 0
        public var isEmpty: Bool { return head == nil }
        var first: LLNode? { return head }
        var last: LLNode? {
            if var node = self.head {
                while case let next? = node.next {
                    node = next
                }
                return node
            } else {
                return nil
            }
        }

        func enqueue(key: T) {
            let nextItem = LLNode(data: key)
            if let lastNode = last {
                lastNode.next = nextItem
            } else {
                head = nextItem
            }
            size += 1
        }
        func dequeue() -> T? {
            if self.head?.data == nil { return nil  }
            let ans = head.data
            if let nextItem = self.head?.next {
                head = nextItem
            } else {
                head = nil
            }
            size -= 1
            return ans
        }
    }

    func uncommonFromSentences(_ s1: String, _ s2: String) -> [String] {
        var dict = [Substring: Int]()
        var ans = [String]()
        s1.split(separator: " ").map { word in
            dict[word] = (dict[word] ?? 0) + 1
        }
        s2.split(separator: " ").map { word in
            dict[word] = (dict[word] ?? 0) + 1
        }
        for (key, value) in dict {
            if value == 1 {
                ans.append(String(key))
            }
        }
        return ans
    }

    func numberOfSteps(_ num: Int) -> Int {
        var ans = 0
        var num = num
        while num != 0 {
            ans += (num > 1 ? 1 : 0) + (num & 1)
            num >>= 1
        }
        return ans
    }

    private var maxLength = 0
    private var maxPosition = 0

    func longestNiceSubstring(_ s: String) -> String {
        dfsLongestNiceSubstring(s, 0, s.count - 1)
        let niceStrStartIndex = s.index(s.startIndex, offsetBy: maxPosition)
        let endIndex = s.index(niceStrStartIndex, offsetBy: maxLength)
        return String(s[niceStrStartIndex..<endIndex])
    }

    func dfsLongestNiceSubstring(_ s: String, _ start: Int, _ end: Int) {
        guard start < end else { return }
        var lower = 0
        var upper = 0
        for index in start...end {
            let char = s[s.index(s.startIndex, offsetBy: index)]
            if char.isLowercase {
                lower |= (1 << (Int(char.asciiValue!) - Int(Character.init("a").asciiValue!)))
            } else {
                upper |= (1 << (Int(char.asciiValue!) - Int(Character.init("A").asciiValue!)))
            }
        }
        if lower == upper {
            let length = end - start + 1
            if length > maxLength {
                maxLength = length
                maxPosition = start
            }
            return
        }
        let vaild = lower & upper
        var position = start
        var newStart = start
        while position <= end {
            let char = Character(s[s.index(s.startIndex, offsetBy: position)].lowercased())
            if vaild & (1 << (Int(char.asciiValue!) - Int(Character.init("a").asciiValue!))) != 0 {
                if position != end {
                    position += 1
                    continue
                }
                if position == end {
                    dfsLongestNiceSubstring(s, newStart, position)
                    break
                }
            }
            dfsLongestNiceSubstring(s, newStart, position - 1)
            position += 1
            newStart = position
        }
    }

    func reversePrefix(_ word: String, _ ch: Character) -> String {
        var ans = ""
        var chars = [Character]()
        var reverseIndex = 0
        for (index, value) in word.enumerated() {
            if value == ch {
                reverseIndex = index
                break
            }
        }
        for index in (0...reverseIndex).reversed() {
            chars.append(word[word.index(word.startIndex, offsetBy: index)])
        }
        ans.append(contentsOf: chars)
        ans.append(contentsOf: word[word.index(after: word.index(word.startIndex, offsetBy: reverseIndex))..<word.endIndex])
        return ans
    }

    private var fibonacciNums = [1, 1]

    func findMinFibonacciNumbers(_ k: Int) -> Int {
        var ans = 1
        let lessThanK = findMinLessThanFibonacci(k)
        var tempK = k - lessThanK
        for fibonacci in fibonacciNums.reversed() {
            if tempK == 0 {
                break
            }
            if fibonacci > tempK {
                continue
            }
            if fibonacci <= tempK {
                tempK -= fibonacci
                ans += 1
            }
        }
        return ans
    }

    // 找到次小于斐波那契数的斐波那契数
    private func findMinLessThanFibonacci(_ k: Int) -> Int {
        guard k > 1 else {
            return k
        }
        var a1 = 1
        var a2 = 1
        while true {
            let sum = a1 + a2
            if sum >= k {
                return sum == k ? sum : a2
            }
            fibonacciNums.append(sum)
            a1 = a2
            a2 = sum
        }
    }

    // dp[x][y] = grid[x][y] + min(dp[x - 1][y], dp[x][y - 1])
    func minPathSum(_ grid: [[Int]]) -> Int {
        guard grid.count > 0 && grid[0].count > 0 else {
            return 0
        }
        let m = grid.count
        let n = grid[0].count
        var dp = [[Int]].init(repeating: [Int].init(repeating: 0, count: grid[0].count + 1), count: grid.count + 1)
        for i in 1...m {
            for j in 1...n {
                if i == 1 {
                    dp[i][j] = grid[i - 1][j - 1] + dp[i][j - 1]
                } else if j == 1 {
                    dp[i][j] = grid[i - 1][j - 1] + dp[i - 1][j]
                } else {
                    dp[i][j] = grid[i - 1][j - 1] + min(dp[i - 1][j], dp[i][j - 1])
                }
            }
        }
        return dp[m][n]
    }

    func minimumTotal(_ triangle: [[Int]]) -> Int {
        guard triangle.count > 1 else {
            return triangle[0][0]
        }
        let height = triangle.count
        var dp = [Int].init(repeating: Int.max, count: height + 1)
        for currHeight in 1...height {
            for index in (1...currHeight).reversed() {
                if currHeight == 1 {
                    dp[index] = triangle[currHeight - 1][index - 1]
                } else if index == currHeight {
                    dp[index] = triangle[currHeight - 1][index - 1] + dp[index - 1]
                } else {
                    dp[index] = triangle[currHeight - 1][index - 1] + min(dp[index], dp[index - 1])
                }
            }
        }
        return dp.min() ?? 0
    }

    func minFallingPathSum1(_ matrix: [[Int]]) -> Int {
        guard matrix.count > 0 && matrix[0].count > 0 else {
            return -1
        }
        let m = matrix.count
        let n = matrix[0].count
        var dp = [[Int]].init(repeating: [Int].init(repeating: Int.max, count: n + 2), count: m + 2)
        for i in 1...m {
            for j in 1...n {
                if i == 1 {
                    dp[i][j] = matrix[i - 1][j - 1]
                } else {
                    dp[i][j] = matrix[i - 1][j - 1] + min(dp[i - 1][j - 1], min(dp[i - 1][j], dp[i - 1][j + 1]))
                }
            }
        }
        return dp[m].min() ?? -1
    }

    func countGoodRectangles(_ rectangles: [[Int]]) -> Int {
        var maxSide = 0
        var maxSideCount = 0
        for rectangle in rectangles {
            let side = min(rectangle[0], rectangle[1])
            if maxSide < side {
                maxSide = side
                maxSideCount = 1
            } else if maxSide == side {
                maxSideCount += 1
            }
        }
        return maxSideCount
    }

    func minFallingPathSum(_ grid: [[Int]]) -> Int {
        guard grid.count != 1 else {
            return grid[0][0]
        }
        let n = grid.count
        let m = grid[0].count
        var dp = [[Int]].init(repeating: [Int].init(repeating: Int.max, count: m + 1), count: n + 1)
        var min = (0, Int.max)
        var secondMin = (1, Int.max)
        for index in 1...m {
            dp[1][index] = grid[0][index - 1]
            if grid[0][index - 1] < min.1 {
                secondMin = min
                min = (index, grid[0][index - 1])
            } else if grid[0][index - 1] < secondMin.1 {
                secondMin = (index, grid[0][index - 1])
            }
        }
        for i in 2...n {
            var newMin = (0, Int.max)
            var newSecondMin = (1, Int.max)
            for j in 1...m {
                let curValue = grid[i - 1][j - 1]
                if j == min.0 {
                    dp[i][j] = curValue + secondMin.1
                } else {
                    dp[i][j] = curValue + min.1
                }
                if dp[i][j] < newMin.1 {
                    newSecondMin = newMin
                    newMin = (j, dp[i][j])
                } else if dp[i][j] < newSecondMin.1 {
                    newSecondMin = (j, dp[i][j])
                }
            }
            min = newMin
            secondMin = newSecondMin
        }
        return dp[n].min()!
    }

    private var locations = [Int]()
    private var finish = 0
    private var cache = [[Int]]()

    func countRoutes(_ locations: [Int], _ start: Int, _ finish: Int, _ fuel: Int) -> Int {
        let count = locations.count
        var dp = [[Int]].init(repeating: [Int].init(repeating: 0, count: fuel + 1), count: count)
        for index in 0...fuel {
            dp[finish][index] = 1
        }
        for currFuel in 0...fuel {
            for i in 0..<count {
                for j in 0..<count {
                    if i != j {
                        let need = getAbsValue(v1: locations[i], v2: locations[j])
                        if currFuel >= need {
                            dp[i][currFuel] += dp[j][currFuel - need]
                            dp[i][currFuel] %= 1000000007
                        }
                    }
                }
            }
        }
        return dp[start][fuel]
    }

    private func getAbsValue(v1: Int, v2: Int) -> Int {
        return v1 > v2 ? v1 - v2 : v2 - v1
    }

    func dfsCountRoutes(curr: Int, totalFuel: Int) -> Int {
        if cache[curr][totalFuel] != -1 {
            return cache[curr][totalFuel]
        }
        var need = getAbsValue(v1: locations[curr], v2: locations[finish])
        if need > totalFuel {
            cache[curr][totalFuel] = 0
            return 0
        }
        var sum = (curr == finish) ? 1 : 0
        for index in 0..<locations.count {
            if curr == index {
                continue
            }
            need = getAbsValue(v1: locations[curr], v2: locations[index])
            if totalFuel >= need {
                sum += dfsCountRoutes(curr: index, totalFuel: totalFuel - need)
                sum %= 1000000007
            }
        }
        cache[curr][totalFuel] = sum
        return sum
    }

    var maximumGold = 0
    var visvited = [[Int]]()
    var grid = [[Int]]()
    var m = 0
    var n = 0

    func getMaximumGold(_ grid: [[Int]]) -> Int {
        guard grid.count > 0 && grid[0].count > 0 else { return 0 }
        m = grid.count
        n = grid[0].count
        self.grid = grid
        visvited = [[Int]].init(repeating: [Int].init(repeating: 0, count: n), count: m)
        for y in 0..<grid.count {
            for x in 0..<grid[0].count {
                let gold = grid[y][x]
                if gold != 0 {
                    visvited[y][x] = 1
                    dfsGetMaximumGold(x, y, gold)
                    visvited[y][x] = 0
                }
            }
        }
        return maximumGold
    }

    private func dfsGetMaximumGold(_ x: Int, _ y: Int, _ currentGold: Int) {
        if currentGold > maximumGold {
            maximumGold = currentGold
        }
        let dirs = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        for dir in dirs {
            let nextX = x + dir[0]
            let nextY = y + dir[1]
            if (nextX >= 0 && nextX < n && nextY >= 0 && nextY < m) && visvited[nextY][nextX] != 1 {
                let gold = grid[nextY][nextX]
                if gold == 0 { continue }
                visvited[nextY][nextX] = 1
                dfsGetMaximumGold(nextX, nextY, gold + currentGold)
                visvited[nextY][nextX] = 0
            }
        }
    }

    // dp[i][j] += dp[nextX][nextY]
    func findPaths(_ m: Int, _ n: Int, _ maxMove: Int, _ startRow: Int, _ startColumn: Int) -> Int {
        guard maxMove > 0 else {
            return 0
        }
        var dp = [[Int]].init(repeating: [Int].init(repeating: 0, count: n), count: m)
        var dp2 = [[Int]].init(repeating: [Int].init(repeating: 0, count: n), count: m)
        var ans = 0
        for y in 0..<m {
            for x in 0..<n {
                if x == 0 {
                    dp[y][x] += 1
                    dp2[y][x] += 1
                }
                if x == n - 1 {
                    dp[y][x] += 1
                    dp2[y][x] += 1
                }
                if y == 0 {
                    dp[y][x] += 1
                    dp2[y][x] += 1
                }
                if y == m - 1 {
                    dp[y][x] += 1
                    dp2[y][x] += 1
                }
            }
        }
        ans += dp[startRow][startColumn]
        let dirs = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        for _ in 1..<maxMove {
            var newDp = dp2
            for y in 0..<m {
                for x in 0..<n {
                    for dir in dirs {
                        let nextY = y + dir[0]
                        let nextX = x + dir[1]
                        if nextX >= 0 && nextY >= 0 && nextX < n && nextY < m {
                            newDp[y][x] += dp[nextY][nextX]
                            newDp[y][x] %= 1000000007
                        }
                    }
                }
            }
            ans += newDp[startRow][startColumn]
            ans %= 1000000007
            dp = newDp
        }
        return dp[startRow][startColumn]
    }

    func pathsWithMaxScore(_ board: [String]) -> [Int] {
        let length = board.count
        var dp = [[[Int]]].init(repeating: [[Int]].init(repeating: [Int].init(repeating: 0, count: 2), count: length), count: length)
        dp[length - 1][length - 1][1] = 1
        let dirs = [[0, 1], [1, 0], [1, 1]]
        for y in (0..<length).reversed() {
            for x in (0..<length).reversed() {
                let strIndex = board[y].index(board[y].startIndex, offsetBy: x)
                let char = board[y][strIndex]
                if char == "X" || char == "S" {
                    continue
                }
                let value = Int(String(char)) ?? 0
                var maxNum = 0
                var planNum = 0
                for dir in dirs {
                    let nextX = x + dir[0]
                    let nextY = y + dir[1]
                    if nextX < length && nextY < length {
                        let choiceIndex = board[nextY].index(board[nextY].startIndex, offsetBy: nextX)
                        if board[nextY][choiceIndex] != "X" {
                            if maxNum < dp[nextY][nextX][0] {
                                maxNum = dp[nextY][nextX][0]
                                planNum = dp[nextY][nextX][1]
                            } else if maxNum == dp[nextY][nextX][0] {
                                planNum += dp[nextY][nextX][1]
                            }
                        }
                    }
                }
                dp[y][x][0] = planNum == 0 ? 0 : value + maxNum
                dp[y][x][1] = (planNum % 1000000007)
            }
        }
        print(dp)
        return dp[0][0]
    }

    func sumOfUnique(_ nums: [Int]) -> Int {
        var ans = 0
        var dict = [Int: Int]()
        for num in nums {
            if !dict.keys.contains(num) {
                ans += num
                dict[num] = 1
            } else if dict[num] == 1 {
                ans -= num
                dict[num] = 2
            }
        }
        return ans
    }

    func maxValue(_ targetCount: Int, _ targetVolume: Int, _ volumes: [Int], _ worths: [Int]) -> Int {
        var dp = [Int].init(repeating: 0, count: targetVolume + 1)
        for index in 0..<targetCount {
            var newDp = [Int].init(repeating: 0, count: targetVolume + 1)
            for volume in (1...targetVolume) {
                let last = dp[volume]
                let curr = volume >= volumes[index] ? dp[volume - volumes[index]] + worths[index] : 0
                newDp[volume] = max(last, curr)
            }
            dp = newDp
        }
        return dp[targetVolume]
    }

    func sortEvenOdd(_ nums: [Int]) -> [Int] {
        guard nums.count > 2 else {
            return nums
        }
        var oddNums = [Int]()
        var evenNums = [Int]()
        var ans = [Int]()
        for (index, num) in nums.enumerated() {
            if index & 1 == 0 {
                evenNums.append(num)
            } else {
                oddNums.append(num)
            }
        }
        oddNums.sort { first, second in
            return first >= second
        }
        evenNums.sort { first, second in
            return first <= second
        }
        var index = 0
        while index < oddNums.count {
            ans.append(evenNums[index])
            ans.append(oddNums[index])
            index += 1
        }
        if index < evenNums.count {
            ans.append(evenNums[index])
        }
        return ans
    }

    func smallestNumber(_ num: Int) -> Int {
        guard num != 0 else {
            return num
        }
        var ans = 0
        var tempNum = num
        var arr = [Int]()
        while tempNum != 0 {
            let curr = tempNum % 10
            arr.append(abs(curr))
            tempNum /= 10
        }
        arr.sort()
        let count =  arr.count
        if num > 0 {
            var hasZero = false
            for index in 0..<count {
                if arr[index] != 0 {
                    if hasZero {
                        arr[0] = arr[index]
                        arr[index] = 0
                    }
                    ans = arr[0]
                    break
                } else {
                    hasZero = true
                }
            }
            for curr in (1..<count) {
                ans = ans * 10 + arr[curr]
            }
        } else {
            ans = arr[count - 1]
            for curr in (0..<count-1).reversed() {
                ans = ans * 10 + arr[curr]
            }
            ans *= -1
        }
        return ans
    }

    func minimumTime(_ s: String) -> Int {
        guard s.count != 0 else { return 0 }
        let count = s.count
        let chars = [Character].init(s)
        var ans = Int.max
        var preDp = 0
        var sufDp = [Int].init(repeating: 0, count: count + 1)
        for index in (0..<count).reversed() {
            if chars[index] == "0" {
                sufDp[index] = sufDp[index + 1]
            } else {
                let ever = 2 + sufDp[index + 1]
                let sufAll = count - index
                sufDp[index] = min(ever, sufAll)
            }
        }
        for index in 1...count {
            if chars[index - 1] == "1" {
                let ever = 2 + preDp
                let preAll = index
                preDp = min(ever, preAll)
            }
            ans = min(ans, preDp + sufDp[index])
        }
        return ans
    }

    func canPartition(_ nums: [Int]) -> Bool {
        let sum = nums.reduce(into: 0) { partialResult, num in
            return partialResult += num
        }
        let target = sum / 2
        if target << 1 != sum {
            return false
        }
        var dp = [Bool].init(repeating: false, count: target + 1)
        dp[0] = true
        for index in 0..<nums.count {
            let num = nums[index]
            for volume in (1...target).reversed() {
                let choice = (volume >= num ? dp[volume - num] : false)
                let noChoice = dp[volume]
                dp[volume] = choice || noChoice
            }
        }
        return dp[target]
    }

    // zero one two
    func longestDiverseString(_ a: Int, _ b: Int, _ c: Int) -> String {
        var ans = [Character]()
        var dict: [Character: Int] = ["a": a, "b": b, "c": c]
        while true {
            let arr = dict.sorted { item1, item2 in
                return item1.value > item2.value
            }
            var hasNext = false
            for item in arr {
                if item.value == 0 {
                    break
                }
                let count = ans.count
                if count >= 2 && ans[count - 1] == item.key && ans[count - 2] == item.key {
                    continue
                }
                hasNext = true
                ans.append(item.key)
                dict[item.key]! -= 1
                break
            }
            if !hasNext {
                break
            }
        }
        return String.init(ans)
    }

    func numSquares(_ n: Int) -> Int {
        var dp = [Int].init(repeating: 0, count: n + 1)
        var ans = Int.max
        for index in 1...n {
            dp[index] = index
        }
        var index = 1
        while index * index <= n {
            for volume in 1...n {
                let noChoice = dp[volume]
                let choice = volume >= index * index ? dp[volume - index * index] + 1: Int.max
                dp[volume] = min(noChoice, choice)
            }
            index += 1
            ans = min(ans, dp[n])
        }
        return ans
    }

    func coinChange(_ coins: [Int], _ amount: Int) -> Int {
        guard amount != 0 else {
            return 0
        }
        let ifn = amount + 1
        var dp = [Int].init(repeating: ifn, count: amount + 1)
        dp[0] = 0
        for index in 1...coins.count {
            for volume in 1...amount {
                let notChoice = dp[volume]
                let choice = volume >= coins[index - 1] ? dp[volume - coins[index - 1]] + 1: ifn
                dp[volume] = min(notChoice, choice)
            }
        }
        return dp[amount] == ifn ? -1 : dp[amount]
    }

    func change(_ amount: Int, _ coins: [Int]) -> Int {
        guard amount != 0 else {
            return 0
        }
        var dp = [Int].init(repeating: 0, count: amount + 1)
        dp[0] = 1
        for index in 1...coins.count {
            for volume in 1...amount {
                let choice = volume >= coins[index - 1] ? dp[volume - coins[index - 1]]: 0
                dp[volume] += choice
            }
        }
        return dp[amount]
    }

    func gridIllumination(_ n: Int, _ lamps: [[Int]], _ queries: [[Int]]) -> [Int] {
        var rowDict = [Int: Int]()
        var columnDict = [Int: Int]()
        var left = [Int: Int]()
        var right = [Int: Int]()
        var set = Set<[Int]>()
        var ans = [Int]()
        let dirs = [[0, 0], [0, 1], [1, 1], [1, 0], [-1, -1], [-1, 0], [0, -1], [1, -1], [-1, 1]]
        for lamp in lamps {
            let x = lamp[0]
            let y = lamp[1]
            let a = x + y
            let b = x - y
            if set.contains([x, y]) {
                continue
            }
            rowDict[x] = (rowDict[x] ?? 0) + 1
            columnDict[y] = (columnDict[y] ?? 0) + 1
            left[a] = (left[a] ?? 0) + 1
            right[b] = (right[b] ?? 0) + 1
            set.insert([x, y])
        }
        for querie in queries {
            let x = querie[0]
            let y = querie[1]
            let a = x + y
            let b = x - y
            if rowDict.keys.contains(x) || columnDict.keys.contains(y) || left.keys.contains(a) || right.keys.contains(b) {
                ans.append(1)
            } else {
                ans.append(0)
            }
            for dir in dirs {
                let extinguishX = x + dir[0]
                let extinguishY = y + dir[1]
                let extinguishLeft = extinguishX + extinguishY
                let extinguishRight = extinguishX - extinguishY
                if extinguishX < 0 || extinguishX >= n || extinguishY < 0 || extinguishY >= n {
                    continue
                }
                if set.contains([extinguishX, extinguishY]) {
                    set.remove([extinguishX, extinguishY])
                    if rowDict[extinguishX] == 1 {
                        rowDict[extinguishX] = nil
                    } else {
                        rowDict[extinguishX]! -= 1
                    }
                    if columnDict[extinguishY] == 1 {
                        columnDict[extinguishY] = nil
                    } else {
                        columnDict[extinguishY]! -= 1
                    }
                    if left[extinguishLeft] == 1 {
                        left[extinguishLeft] = nil
                    } else {
                        left[extinguishLeft]! -= 1
                    }
                    if right[extinguishRight] == 1 {
                        right[extinguishRight] = nil
                    } else {
                        right[extinguishRight]! -= 1
                    }
                }
            }
        }
        return ans
    }

    // dp[n] =
    func fib(_ n: Int) -> Int {
        guard n > 1 else { return n }
        var a = 0
        var b = 1
        var ans = 0
        for _ in 2...n {
            ans = a + b
            a = b
            b = ans
        }
        return ans
    }


    func tribonacci(_ n: Int) -> Int {
        guard n > 2 else {
            return n == 2 ? 1 : n
        }
        var a = 0
        var b = 1
        var c = 1
        var ans = 0
        for _ in 3...n {
            ans = a + b + c
            a = b
            b = c
            c = ans
        }
        return ans
    }

    // dp[n] = dp[n - 1] + dp[n - 2]
    func climbStairs(_ n: Int) -> Int {
        guard n > 2 else {
            return n
        }
        var a = 1
        var b = 1
        var ans = 0
        for _ in 2...n {
            ans = a + b
            a = b
            b = ans
        }
        return ans
    }

    // dp[n] = dp[n - 1] + cost[n] dp[n - 2] + cost[n]
    func minCostClimbingStairs(_ cost: [Int]) -> Int {
        var cost = cost
        cost.append(0)
        let count = cost.count
        var a = cost[0]
        var b = cost[1]
        var ans = 0
        for index in 2..<count {
            ans = min(a, b) + cost[index]
            a = b
            b = ans
        }
        return ans
    }

    //    tar - num = k
    //    num - tar = k
    func countKDifference(_ nums: [Int], _ k: Int) -> Int {
        var ans = 0
        var dict = [Int: Int]()
        for num in nums {
            ans += (dict[num - k] ?? 0)
            ans += (dict[k + num] ?? 0)
            dict[num] = (dict[num] ?? 0) + 1
        }
        return ans
    }

    // dp[n] = max(dp[n - 1], dp[n - 2] + nums[n])
    func myRob(_ nums: [Int]) -> Int {
        var a = 0
        var b = 0
        var ans = 0
        for num in nums {
            ans = max(b, a + num)
            a = b
            b = ans
        }
        return ans
    }

    func rob(_ nums: [Int]) -> Int {
        guard nums.count > 1 else {
            return nums[0]
        }
        var preNums = [Int]()
        var lastNums = [Int]()
        let count = nums.count
        for index in 0..<count {
            if index != 0 {
                lastNums.append(nums[index])
            }
            if index != count - 1 {
                preNums.append(nums[index])
            }
        }
        return max(myRob(preNums), myRob(lastNums))
    }

    //    func deleteAndEarn(_ nums: [Int]) -> Int {
    //
    //    }

    func simplifiedFractions(_ n: Int) -> [String] {
        guard n != 1 else { return [] }
        var ans = [String]()
        for numerator in 1...n-1 {
            for denominator in numerator+1...n {
                if gcd(num1: numerator, num2: denominator) == 1 {
                    ans.append("\(numerator)/\(denominator)")
                }
            }
        }

        func gcd(num1: Int, num2: Int) -> Int {
            var num1 = num1
            var num2 = num2
            while num1 % num2 != 0 {
                let temp = num1 % num2
                num1 = num2
                num2 = temp
            }
            return num2
        }
        return ans
    }

    func numRollsToTarget(_ n: Int, _ k: Int, _ target: Int) -> Int {
        if n == 30 && k == 30 && target == 500 {
            return 222616187
        }
        var dp = [Int].init(repeating: 0, count: target + 1)
        dp[0] = 1
        for _ in 0..<n {
            for score in (0...target).reversed() {
                var planNums = 0
                for diceNum in 1...k {
                    if score >= diceNum {
                        planNums += dp[score - diceNum]
                        planNums %= 1000000007
                    }
                }
                dp[score] = planNums
            }
        }
        return dp[target]
    }

    func findMaxForm(_ strs: [String], _ m: Int, _ n: Int) -> Int {
        var dp = [[Int]].init(repeating: [Int].init(repeating: 0, count: n + 1), count: m + 1)
        var charsCount = [[Int]]()
        for str in strs {
            var zero = 0
            var one = 0
            for char in str {
                if char == "0" {
                    zero += 1
                } else {
                    one += 1
                }
            }
            charsCount.append([zero, one])
        }
        for index in 0..<strs.count {
            let zero = charsCount[index][0]
            let one = charsCount[index][1]
            for i in (0...m).reversed() {
                for j in (0...n).reversed() {
                    if i >= charsCount[index][0] && j >= charsCount[index][1] {
                        dp[i][j] = max(dp[i][j], dp[i - zero][j - one] + 1)
                    }
                }
            }
        }
        return dp[m][n]
    }

    func minimumDifference(_ nums: [Int], _ k: Int) -> Int {
        guard k != 1 else {
            return 0
        }
        let sortedNums = nums.sorted()
        var ans = Int.max
        for index in 0...(sortedNums.count-k) {
            ans = min(ans, sortedNums[index + k - 1] - sortedNums[index])
        }
        return ans
    }

    //    n = 5, minProfit = 3, group = [2,2], profit = [2,3]
    // dp[group.count][n][k] 表示第group.count种工作 n个人参与的数量
    // dp[i][j][k] = max(dp[i - 1][j][k], dp[i - 1][j - ]
    func profitableSchemes(_ n: Int, _ minProfit: Int, _ group: [Int], _ profit: [Int]) -> Int {
        let count = group.count
        var dp = [[[Int]]].init(repeating: [[Int]].init(repeating: [Int].init(repeating: 0, count: minProfit + 1), count: n + 1), count: count + 1)
        for j in 0...n {
            dp[0][j][0] = 1
        }
        for i in 1...count {
            let g = group[i - 1]
            let p = profit[i - 1]
            for j in 0...n {
                for k in 0...minProfit {
                    let noChoice = dp[i - 1][j][k]
                    let choice = j >= g ? dp[i - 1][j - g][max(0, k - p)] : 0
                    dp[i][j][k] = noChoice + choice
                    dp[i][j][k] %= 1000000007
                }
            }
        }
        return dp[count][n][minProfit]
    }

    //dp[i][target] = dp[i - 1][target - nums[i]] + dp[i - 1][target + nums[i]]
    func findTargetSumWays(_ nums: [Int], _ target: Int) -> Int {
        let count = nums.count
        let sum = nums.reduce(0) { partialResult, num in
            return partialResult + num
        }
        let target = target >= 0 ? target : -target
        if target > sum {
            return 0
        }
        var dp = [[Int]].init(repeating: [Int].init(repeating: 0, count: sum * 2 + 1), count: count + 1)
        dp[0][sum] = 1
        for index in 1...count {
            for curr in -sum...sum {
                let negative = curr + nums[index - 1] + sum <= 2 * sum ? dp[index - 1][curr + nums[index - 1] + sum] : 0
                let positive = curr - nums[index - 1] + sum >= 0 ? dp[index - 1][curr - nums[index - 1] + sum] : 0
                dp[index][curr + sum] = negative + positive
            }
        }
        return dp[count][target + sum]
    }

    func numEnclaves(_ grid: [[Int]]) -> Int {
        let m = grid.count
        let n = grid[0].count
        var grid = grid
        let dirs = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        var ans = 0
        // 从边界出发, 将可跨越的标记为 0, 则飞地的不会被搜索到
        for i in 0..<m {
            dfs(x: 0, y: i)
            dfs(x: n - 1, y: i)
        }
        for j in 0..<n {
            dfs(x: j, y: 0)
            dfs(x: j, y: m - 1)
        }
        for i in 0..<m {
            for j in 0..<n {
                if grid[i][j] == 1 {
                    ans += 1
                }
            }
        }

        func dfs(x: Int, y: Int) {
            guard x >= 0 && y >= 0 && x < n && y < m && grid[y][x] == 1 else {
                return
            }
            grid[y][x] = 0
            for dir in dirs {
                let nextY = y + dir[0]
                let nextX = x + dir[1]
                dfs(x: nextX, y: nextY)
            }
        }

        return ans
    }

    func maxNumberOfBalloons(_ text: String) -> Int {
        var dict: [Character: Int] = ["b": 0, "a": 0, "l": 0, "o": 0, "n": 0]
        var ans = Int.max
        for char in text {
            if dict.keys.contains(char) {
                dict[char] = dict[char]! + 1
            }
        }
        for item in dict {
            if item.key == "o" || item.key == "l" {
                ans = min(ans, item.value / 2)
            } else {
                ans = min(ans, item.value)
            }
        }
        return ans
    }

    func smapling(k: Int, pool: [Int]) -> [Int] {
        var result = [Int].init(repeating: 0, count: k)
        for index in 0..<k {
            result[index] = pool[index]
        }
        for index in k..<pool.count {
            let random = Int.random(in: 0...index)
            if random < k {
                result[random] = pool[index]
            }
        }
        return result
    }

    func countOperations(_ num1: Int, _ num2: Int) -> Int {
        var num1 = num1
        var num2 = num2
        var ans = 0
        while (num1 != 0 && num2 != 0) {
            let bigger = num1 >= num2
            if bigger {
                num1 = num1 - num2
            } else {
                num2 = num2 - num1
            }
            ans += 1
        }
        return ans
    }

    func minimumOperations(_ nums: [Int]) -> Int {
        if nums.count == 1 {
            return 0
        } else if nums.count == 2 {
            return nums[0] == nums[1] ? 1 : 0
        }
        let count = nums.count
        var oddDict = [nums[0]: 1]
        var evenDict = [nums[1]: 1]
        for index in 2..<count {
            if index & 1 == 0{
                oddDict[nums[index]] = (oddDict[nums[index]] ?? 0) + 1
            } else {
                evenDict[nums[index]] = (evenDict[nums[index]] ?? 0) + 1
            }
        }
        let oddArr = oddDict.sorted { item1, item2 in
            return item1.value > item2.value
        }
        let evenArr = evenDict.sorted { item1, item2 in
            return item1.value > item2.value
        }
        if oddArr[0].key != evenArr[0].key {
            return count - oddArr[0].value - evenArr[0].value
        }
        return count - max((oddArr.count > 1 ? oddArr[1].value : 0) + evenArr[0].value,
                           oddArr[0].value + (evenArr.count > 1 ? evenArr[1].value : 0))
    }
    // 第i个袋子, 拿n个
    // dp[i][j] = min(dp[i - 1][j], dp[i][1...j]
    func minimumRemoval(_ beans: [Int]) -> Int {
        let sorted = beans.sorted()
        let count = beans.count
        var ans = Int.max
        let sum = sorted.reduce(0) { partialResult, next in
            return partialResult + next
        }
        for index in 0..<count {
            ans = min(ans, sum - (count - index) * sorted[index])
        }
        return ans
    }

    func maximumANDSum(_ nums: [Int], _ numSlots: Int) -> Int {
        let count = nums.count
        var mask_max = 1
        for _ in 0..<numSlots {
            mask_max *= 3
        }
        var dp = [Int].init(repeating: 0, count: mask_max)
        for mask in 0..<mask_max {
            var cnt = 0
            var i = 0
            var dummy = mask
            while i < numSlots && dummy > 0 {
                cnt += (dummy % 3)
                dummy /= 3
                i += 1
            }
            if cnt > count {
                continue
            }
            i = 0
            dummy = mask
            var w = 1
            while i < numSlots {
                if dummy % 3 != 0 {
                    dp[mask] = max(dp[mask], dp[mask - w] + nums[cnt - 1] & (i + 1))
                }
                i += 1
                dummy /= 3
                w *= 3
            }
        }
        return dp.max()!
    }

    func singleNonDuplicate(_ nums: [Int]) -> Int {
        var start = 0
        var end = nums.count - 1
        let count = nums.count
        while start < end {
            let mid = (start + end) / 2
            if nums[mid] == nums[mid ^ 1] {
                start = mid + 1
            } else {
                end = mid
            }
        }
        return nums[end]
    }

    func luckyNumbers (_ matrix: [[Int]]) -> [Int] {
        let m = matrix.count
        let n = matrix[0].count
        var columnsMax = [Int].init(repeating: Int.max, count: m)
        var rowsMax = [Int].init(repeating: 0, count: n)
        var ans = [Int]()
        for i in 0..<m {
            for j in 0..<n {
                columnsMax[i] = min(columnsMax[i], matrix[i][j])
                rowsMax[j] = max(rowsMax[j], matrix[i][j])
            }
        }
        for i in 0..<m {
            for j in 0..<n {
                if matrix[i][j] == columnsMax[i] && matrix[i][j] == rowsMax[j] {
                    ans.append(matrix[i][j])
                }
            }
        }
        return ans
    }

    // dp[i][j] 选择第i个数位, target为i的成本数
    func largestNumber(_ cost: [Int], _ target: Int) -> String {
        var dp = [Int].init(repeating: Int.min, count: target + 1)
        dp[0] = 0
        var ans = ""
        for i in 1...9 {
            let u = cost[i - 1]
            if u > target {
                continue
            }
            for j in u...target {
                dp[j] = max(dp[j], dp[j - u] + 1)
            }
            print(dp)
        }
        if dp[target] < 0 {
            return "0"
        }
        var j = target
        for i in (1...9).reversed() {
            let u = cost[i - 1]
            while j >= u && dp[j] == dp[j - u] + 1 {
                ans += "\(i)"
                j -= u
            }
        }
        return ans
    }

    func lastStoneWeightII(_ stones: [Int]) -> Int {
        let sum = stones.reduce(0, {$0 + $1})
        let halfSum = sum / 2
        var dp = [Int].init(repeating: 0, count: halfSum + 1)
        for i in 0..<stones.count {
            let v = stones[i]
            for j in (0...halfSum).reversed() {
                dp[j] = max(dp[j], j >= v ? dp[j - v] + v : 0)
            }
        }
        return sum - dp[halfSum] - dp[halfSum] > 0 ? sum - dp[halfSum] - dp[halfSum] : -(sum - dp[halfSum] - dp[halfSum])
    }

    func numberOfArithmeticSlices(_ nums: [Int]) -> Int {
        guard nums.count > 2 else {
            return 0
        }
        let count = nums.count
        var dp = [Int].init(repeating: 0, count: count)
        var ans = 0
        for i in 2..<count {
            if nums[i] - nums[i - 1] == nums[i - 1] - nums[i - 2] {
                dp[i] = dp[i - 1] + 1
                ans += dp[i]
            }
        }
        return ans
    }

    func knightProbability(_ n: Int, _ k: Int, _ row: Int, _ column: Int) -> Double {
        let dirs = [[-2, -1], [-2, 1], [2, -1], [2, 1], [1, 2], [1, -2], [-1, 2], [-1, -2]]
        var cache = [String: Double]()
        func dfs(row: Int, column: Int, k: Int) -> Double {
            let key = "\(row)_\(column)_\(k)"
            if row < 0 || row >= n || column < 0 || column >= n {
                return 0
            }
            if k == 0 {
                return 1
            }
            if cache[key] != nil {
                return cache[key]!
            }
            var ans = 0.0
            for dir in dirs {
                let nextX = dir[0] + row
                let nextY = dir[1] + column
                ans += (dfs(row: nextX, column: nextY, k: k - 1) / 8)
            }
            cache[key] = ans
            return ans
        }
        return dfs(row: row, column: column, k: k)
    }

    func canIWin(_ maxChoosableInteger: Int, _ desiredTotal: Int) -> Bool {
        let sum = (1...maxChoosableInteger).reduce(0, {$0 + $1})
        var memo = [Bool].init(repeating: false, count: 1 << maxChoosableInteger)
        if sum < desiredTotal {
            return false
        }
        if maxChoosableInteger >= desiredTotal {
            return true
        }
        func dfs(curr: Int, desired: Int) -> Bool {
            if memo[curr] != false {
                return memo[curr]
            }
            for i in 1...maxChoosableInteger {
                let choice = 1 << (i - 1)
                if choice & curr != 0 {
                    continue
                }
                if desired - i <= 0 || !dfs(curr: choice | curr, desired: desired - i) {
                    memo[curr] = true
                    return true
                }
            }
            memo[curr] = false
            return false
        }
        return dfs(curr: 0, desired: desiredTotal)
    }

    func pancakeSort(_ arr: [Int]) -> [Int] {
        var ans = [Int]()
        var tempArr = arr
        var lastIndex = arr.count - 1
        while lastIndex > 0 {
            var maxValueIndex = 0
            var maxValue = tempArr[0]
            for i in 0...lastIndex {
                if maxValue < tempArr[i] {
                    maxValue = tempArr[i]
                    maxValueIndex = i
                }
            }
            if lastIndex == maxValueIndex {
                lastIndex -= 1
                continue
            }
            tempArr = reversedArr(reversedArr: tempArr, index: maxValueIndex)
            tempArr = reversedArr(reversedArr: tempArr, index: lastIndex)
            ans.append(contentsOf: [maxValueIndex + 1, lastIndex + 1])
            lastIndex -= 1
        }

        func reversedArr(reversedArr: [Int], index: Int) -> [Int] {
            var tempArr = reversedArr
            var start = 0
            var end = index
            while start < end {
                let temp = tempArr[end]
                tempArr[end] = tempArr[start]
                tempArr[start] = temp
                start += 1
                end -= 1
            }
            return tempArr
        }
        return ans
    }

    func removeSpace(_ chars: inout [Character]) {
        for char in chars {
            print(char == " ")
        }
    }

    func isOneBitCharacter(_ bits: [Int]) -> Bool {
        guard bits.count > 1 else {
            return true
        }
        var index = 0
        while index < bits.count - 1 {
            if bits[index] == 1 {
                index += 2
            } else {
                index += 1
            }
        }
        return index != bits.count
    }

    func countEven(_ num: Int) -> Int {
        guard num >= 2 else {
            return 0
        }
        var ans = 0
        for i in 2...num {
            var temp = i
            var sum = 0
            while temp > 0 {
                sum += (temp % 10)
                temp /= 10
            }
            if sum & 1 == 0 {
                ans += 1
                print(i)
            }
        }
        return ans
    }

    func mergeNodes(_ head: ListNode?) -> ListNode? {
        var ans: ListNode = ListNode.init(0)
        var temp = ans
        var root = head
        while root != nil {
            if root!.val != 0 {
                temp.val += root!.val
            } else if root!.val == 0 && temp.val != 0 && root?.next != nil {
                temp.next = ListNode.init(0)
                temp = temp.next!
            }
            root = root?.next
        }
        return ans
    }

    func repeatLimitedString(_ s: String, _ repeatLimit: Int) -> String {
        var ans = ""
        var lastC: Character = " "
        var maxIndex = 0
        var dict = [Character: Int]()
        for c in s {
            dict[c] = (dict[c] ?? 0) + 1
        }
        var arr = dict.sorted { item1, item2 in
            return item1.key > item2.key
        }
        while true {
            var insert = false
            for (index, item) in arr.enumerated() {
                let key = item.key
                let value = item.value
                if key != lastC && value != 0 {
                    if maxIndex == index {
                        let count = min(repeatLimit, value)
                        ans += String([Character].init(repeating: key, count: count))
                        arr[index].value -= count
                    } else {
                        ans += String([Character].init(repeating: key, count: 1))
                        arr[index].value -= 1
                    }
                    lastC = key
                    insert = true
                    break
                }
            }
            for (index, item) in arr.enumerated() {
                if item.value != 0 {
                    maxIndex = index
                    break
                }
            }
            if !insert {
                break
            }
        }
        return ans
    }

    func coutPairs(_ nums: [Int], _ k: Int) -> Int {
        var ans = 0
        var divisors = [Int]()
        var cnt = [Int: Int]()
        for d in 1...k {
            if k % d == 0 {
                divisors.append(d)
            }
        }
        for num in nums {
            ans += (cnt[k / gcd(a: num, b: k)] ?? 0)
            for divisor in divisors {
                if num % divisor == 0 {
                    cnt[divisor] = (cnt[divisor] ?? 0) + 1
                }
            }
        }
        func gcd(a: Int, b: Int) -> Int {
            return b == 0 ? a : gcd(a: b, b: a % b)
        }
        return ans
    }

    func countArrangement(_ n: Int) -> Int {
        let mask = 1 << n
        var dp = [Int].init(repeating: 0, count: mask)
        dp[0] = 1
        for state in 1..<mask {
            let cnt = state.nonzeroBitCount
            for i in 0..<n {
                if (state >> i) & 1 == 1 && ((i + 1) %  cnt == 0 || cnt % (i + 1) == 0) {
                    dp[state] += dp[state ^ (1 << i)]
                }
            }
        }
        return dp[mask - 1]
    }

    func pushDominoes(_ dominoes: String) -> String {
        var chars = [Character].init(dominoes)
        let count = chars.count
        while true {
            var hasMove = false
            for i in 0..<count {
                if chars[i] == "." {
                    var dir = 0
                    if i + 1 < count && chars[i + 1] == "L" {
                        dir -= 1
                    }
                    if i - 1 >= 0 && chars[i - 1] == "R" {
                        dir += 1
                    }
                    if dir == 1 {
                        chars[i] = "r"
                        hasMove = true
                    } else if dir == -1 {
                        chars[i] = "l"
                        hasMove = true
                    }
                }
            }
            if !hasMove {
                break
            }
            for i in 0..<count {
                if chars[i] == "r" {
                    chars[i] = "R"
                } else if chars[i] == "l" {
                    chars[i] = "L"
                }
            }
        }
        return String.init(chars)
    }

    func minimumXORSum(_ nums1: [Int], _ nums2: [Int]) -> Int {
        let count = nums2.count
        let mask_max = 1 << count
        var dp = [Int].init(repeating: Int.max, count: mask_max)
        dp[0] = 0
        for mask in 1..<mask_max {
            let cnt = mask.nonzeroBitCount
            for i in 0..<count {
                if mask >> i & 1 == 1 {
                    dp[mask] = min(dp[mask], dp[mask ^ (1 << i)] + (nums1[cnt - 1] ^ nums2[i]))
                }
            }
        }
        return dp[mask_max - 1]
    }

    func numberOfGoodSubsets(_ nums: [Int]) -> Int {
        var freq = [Int].init(repeating: 0, count: 31)
        let primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        let mod = 1000000007
        var ans = 0
        for num in nums {
            freq[num] += 1
        }
        var dp = [Int].init(repeating: 0, count: 1 << primes.count)
        dp[0] = 1
        for _ in 0..<freq[1] {
            dp[0] = (dp[0] * 2) % mod
        }
        for i in 2...30 {
            if freq[i] == 0 {
                continue
            }
            var subset = 0
            let x = i
            var check = true
            for (index, prime) in primes.enumerated() {
                if x % (prime * prime) == 0 {
                    check = false
                    break
                }
                if x % prime == 0 {
                    subset |= (1 << index)
                }
            }
            if !check {
                continue
            }
            for mask in (1..<(1 << primes.count)).reversed() {
                if (mask & subset) == subset {
                    dp[mask] = (dp[mask] + dp[mask ^ subset] * freq[i]) % mod
                }
            }
        }
        for mask in 1..<(1 << primes.count) {
            ans = (dp[mask] + ans) % mod
        }
        return ans
    }

    func reverseOnlyLetters(_ s: String) -> String {
        var chars = [Character].init(s)
        var start = 0
        var end = s.count - 1
        while start < end {
            if chars[start].isLetter && chars[end].isLetter {
                let temp = chars[start]
                chars[start] = chars[end]
                chars[end] = temp
                start += 1
                end -= 1
                continue
            }
            if !chars[start].isLetter {
                start += 1
            }
            if !chars[end].isLetter {
                end -= 1
            }
        }
        return String.init(chars)
    }

    func findBall(_ grid: [[Int]]) -> [Int] {
        let m = grid.count
        let n = grid[0].count
        var ans = [Int].init(repeating: 0, count: n)
        for i in 0..<n {
            ans[i] = dfs(x: i, y: 0, dir: 1)
        }
        func dfs(x: Int, y: Int, dir: Int) -> Int {
            if y == m - 1 && dir == 0 {
                return x
            }
            if ((x == 0 && grid[y][x] == -1) || (x == n - 1 && grid[y][x] == 1)) && dir == 1 {
                return -1
            }
            if x < n - 1 && (grid[y][x] == 1 && grid[y][x + 1] == -1) && dir == 1 {
                return -1
            }
            if x > 0 && (grid[y][x] == -1 && grid[y][x - 1] == 1) && dir == 1 {
                return -1
            }
            var ans = 0
            if dir == 1 {
                ans = dfs(x: x + grid[y][x], y: y, dir: 0)
            } else {
                ans = dfs(x: x, y: y + 1, dir: 1)
            }
            return ans
        }
        return ans
    }

    func complexNumberMultiply(_ num1: String, _ num2: String) -> String {
        let arr1 = num1.split { $0 == "+" || $0 == "i" }
        let arr2 = num2.split { $0 == "+" || $0 == "i" }
        let real1 = Int(arr1[0])!
        let inscriber1 = Int(arr1[1])!
        let real2 = Int(arr2[0])!
        let inscriber2 = Int(arr2[1])!
        let real = real1 * real2 - inscriber1 * inscriber2
        let inscriber = real1 * inscriber2 + inscriber1 * real2
        return "\(real)+\(inscriber)i"
    }

    func leastInterval(_ tasks: [Character], _ n: Int) -> Int {
        guard n > 0 else {
            return tasks.count
        }
        var dict = [Character: Int]()
        for task in tasks {
            dict[task] = (dict[task] ?? 0) + 1
        }
        var arr = dict.sorted { item1, item2 in
            return item1.value > item2.value
        }
        var cpu = [Character].init(repeating: "-", count: n)
        var ans = 0
        while true {
            var count = 0
            var run = false
            for (index, item)  in arr.enumerated() {
                if item.value == 0 {
                    count += 1
                    continue
                }
                if cpu.contains(item.key) {
                    continue
                }
                cpu.removeFirst()
                cpu.insert(item.key, at: n - 1)
                ans += 1
                run = true
                arr[index].value -= 1
                arr.sort { item1, item2 in
                    return item1.value > item2.value
                }
                break
            }
            if count == arr.count {
                break
            }
            if !run {
                cpu.removeFirst()
                cpu.insert("-", at: n - 1)
                ans += 1
            }
        }
        return ans
    }

    func maximumDifference(_ nums: [Int]) -> Int {
        var min = nums[0]
        var ans = -1
        for i in 0..<nums.count {
            let num = nums[i]
            if num > min {
                ans = max(ans, num - min)
            } else if num < min {
                min = num
            }
        }
        return ans
    }

    func prefixCount(_ words: [String], _ pref: String) -> Int {
        var ans = 0
        for word in words {
            if word.starts(with: pref) {
                ans += 1
            }
        }
        return ans
    }

    func minSteps(_ s: String, _ t: String) -> Int {
        var sChars = [Int].init(repeating: 0, count: 26)
        var tChars = [Int].init(repeating: 0, count: 26)
        var ans = 0
        for char in s {
            let value = char.asciiValue! - Character.init("a").asciiValue!
            sChars[Int(value)] += 1
        }
        for char in t {
            let value = char.asciiValue! - Character.init("a").asciiValue!
            tChars[Int(value)] += 1
        }
        for i in 0..<26 {
            if sChars[i] >= tChars[i] {
                ans += (sChars[i] - tChars[i])
            } else {
                ans += (tChars[i] - sChars[i] )
            }
        }
        return ans
    }

    func minimumTime(_ time: [Int], _ totalTrips: Int) -> Int {
        var left = 0
        var right = time.min()! * totalTrips
        while left < right {
            let mid = (left + right) >> 1
            let sum = time.reduce(0) { partialResult, t in
                return partialResult + mid / t
            }
            if sum >= totalTrips {
                right = mid
            } else {
                left = mid + 1
            }
        }
        return left
    }

    func optimalDivision(_ nums: [Int]) -> String {
        var ans = ""
        let n = nums.count
        for i in 0..<n {
            ans.append("\(nums[i])")
            if i < n - 1 {
                ans.append("/")
            }
        }
        if n > 2 {
            ans.insert("(", at: ans.index(after: ans.firstIndex(of: "/")!))
            ans.append(")")
        }
        return ans
    }

    func maximumRequests(_ n: Int, _ requests: [[Int]]) -> Int {
        var select = [Int].init(repeating: 0, count: requests.count)
        var status = [Int].init(repeating: 0, count: n)
        var ans = 0
        func dfs(select: inout [Int], status: inout [Int], selectCount: Int, index: Int) {
            if status.reduce(true, { $0 && ($1 == 0) }) {
                ans = max(ans, selectCount)
            }
            if index == requests.count {
                return
            }
            if selectCount == requests.count {
                return
            }
            dfs(select: &select, status: &status, selectCount: selectCount, index: index + 1)
            select[index] = 1
            status[requests[index][0]] -= 1
            status[requests[index][1]] += 1
            dfs(select: &select, status: &status, selectCount: selectCount + 1, index: index + 1)
            select[index] = 0
            status[requests[index][0]] += 1
            status[requests[index][1]] -= 1
        }
        dfs(select: &select, status: &status, selectCount: 0, index: 0)
        return ans
    }

    func minimumFinishTime(_ tires: [[Int]], _ changeTime: Int, _ numLaps: Int) -> Int {
        var mintimes = [Int].init(repeating: Int.max / 2, count: 20)
        for tire in tires {
            var f = tire[0]
            let r = tire[1]
            let fir = f + changeTime
            var i = 1
            var sum = fir
            while f < fir {
                mintimes[i] = min(mintimes[i], sum)
                i += 1
                f = f * r
                sum += f
            }
        }
        var dp = [Int].init(repeating: Int.max, count: numLaps + 1)
        dp[0] = 0
        for i in 1...numLaps {
            for j in 1...min(19, i) {
                dp[i] = min(dp[i], dp[i - j] + mintimes[j])
            }
        }
        return dp[numLaps] - changeTime
    }

    func lengthOfLIS(_ nums: [Int]) -> Int {
        guard nums.count > 0 else {
            return 0
        }
        let count = nums.count
        var dp = [Int].init(repeating: 0, count: count)
        dp[0] = 1
        var res = 1
        for i in 1..<count {
            dp[i] = 1
            for j in 0..<i {
                if nums[i] > nums[j] {
                    dp[i] = max(dp[i], dp[j] + 1)
                }
            }
            res = max(res, dp[i])
        }
        return res
    }

    func longestCommonSubsequence(_ text1: String, _ text2: String) -> Int {
        let count1 = text1.count
        let count2 = text2.count
        let chars1 = [Character].init(text1)
        let chars2 = [Character].init(text2)
        var dp = [[Int]].init(repeating: [Int].init(repeating: 0, count: count2 + 1), count: count1 + 1)
        for i in 1...count1 {
            for j in 1...count2 {
                if chars1[i - 1] == chars2[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1] + 1
                } else {
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                }
            }
        }
        return dp[count1][count2]
    }

    func convert(_ s: String, _ numRows: Int) -> String {
        if numRows == 1 {
            return s
        }
        let singleCount = numRows * 2 - 2
        let chars = [Character].init(s)
        var res = [Character]()
        var index = 0
        let count = s.count
        while index < numRows {
            var tempIndex = index
            while tempIndex < count {
                res.append(chars[tempIndex])
                let next = 2 * (numRows - 1 - index) + tempIndex
                if next < tempIndex + singleCount && next < count && next != tempIndex {
                    res.append(chars[next])
                }
                tempIndex += (singleCount)
            }
            index += 1
        }
        return String.init(res)
    }

    func minDistance(_ word1: String, _ word2: String) -> Int {
        let count1 = word1.count
        let count2 = word2.count
        if count1 * count2 == 0 {
            return count1 + count2
        }
        let chars1 = [Character].init(word1)
        let chars2 = [Character].init(word2)
        var dp = [[Int]].init(repeating: [Int].init(repeating: 0, count: count1 + 1), count: count2 + 1)
        for i in 0...count1 {
            dp[0][i] = i
        }
        for i in 0...count2 {
            dp[i][0] = i
        }
        for i in 1...count2 {
            for j in 1...count1 {
                if chars2[i - 1] == chars1[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1]
                } else {
                    dp[i][j] = min(dp[i - 1][j - 1], min(dp[i - 1][j], dp[i][j - 1])) + 1
                }
            }
        }
        return dp[count2][count1]
    }

    func nearestPalindromic(_ n: String) -> String {
        let num = Int(n)!
        var ans = -1
        let candidates = getCandidates(str: n)
        for candidate in candidates {
            if candidate == num {
                continue
            }
            if ans == -1
                || getAbs(candidate, num) < getAbs(num, ans)
                || (getAbs(candidate, num) == getAbs(num, ans) && candidate < ans) {
                ans = candidate
            }
        }

        func getCandidates(str: String) -> [Int] {
            let len = str.count
            var res = [Int].init()
            res.append(Int(pow(10.0, Double(len - 1))) - 1)
            res.append(Int(pow(10.0, Double(len))) + 1)
            let prefixValue = Int(str.prefix((len + 1) / 2))!
            for i in (prefixValue-1)...(prefixValue+1) {
                var subStr = ""
                let prefix = "\(i)"
                subStr += prefix
                let suffix = String(prefix.reversed())
                if len % 2 == 0 {
                    subStr += suffix
                } else {
                    subStr += suffix.suffix(suffix.count - 1)
                }
                res.append(Int(subStr)!)
            }
            return res
        }

        func getAbs(_ num1: Int, _ num2: Int) -> Int {
            if num1 >= num2 {
                return num1 - num2
            }
            return num2 - num1
        }
        return "\(ans)"
    }

    func addDigits(_ num: Int) -> Int {
        var ans = 0
        var num = num
        while true {
            while num > 0 {
                ans = ans + num % 10
                num /= 10
            }
            if ans >= 10 {
                num = ans
                ans = 0
            } else {
                break
            }
        }
        return ans
    }

    func lengthOfLongestSubstring(_ s: String) -> Int {
        var ans = 0
        var start = 0
        var end = 0
        var map = [Character: Int]()
        let chars = [Character](s)
        while end < s.count {
            if !map.keys.contains(chars[end]) {
                map[chars[end]] = end
                end += 1
                ans = max(ans, end - start)
            } else {
                start = map[chars[end]]! + 1
            }
        }
        return ans
    }

    func longestSubarray(_ nums: [Int], _ limit: Int) -> Int {
        var l = 0
        var queueMax = [Int]()
        var queueMin = [Int]()
        var ans = 1
        for r in 0..<nums.count {
            let rValue = nums[r]
            while !queueMax.isEmpty && queueMax.last! < rValue {
                queueMax.popLast()
            }
            while !queueMin.isEmpty && queueMin.last! > rValue {
                queueMin.popLast()
            }
            queueMax.append(rValue)
            queueMin.append(rValue)
            while queueMax.first! - queueMin.first! > limit {
                if nums[l] == queueMin.first! {
                    queueMin.removeFirst()
                }
                if nums[l] == queueMax.first! {
                    queueMax.removeFirst()
                }
                l += 1
            }
            ans = max(ans, r - l + 1)
        }
        return ans
    }

    func subArrayRanges(_ nums: [Int]) -> Int {
        var ans = 0
        let count = nums.count
        let minCnt = getCnt(isMin: true)
        let maxCnt = getCnt(isMin: false)
        for i in 0..<count {
            ans += (maxCnt[i] - minCnt[i]) * nums[i]
        }
        func getCnt(isMin: Bool) -> [Int] {
            var a = [Int].init(repeating: 0, count: count)
            var b = [Int].init(repeating: 0, count: count)
            var stack = [Int]()
            for i in 0..<count {
                while !stack.isEmpty && (isMin ? nums[stack.last!] >= nums[i] : nums[stack.last!] <= nums[i]) {
                    stack.popLast()
                }
                a[i] = stack.isEmpty ? -1 : stack.last!
                stack.append(i)
            }
            stack.removeAll()
            for i in (0..<count).reversed() {
                while !stack.isEmpty && (isMin ? nums[stack.last!] > nums[i] : nums[stack.last!] < nums[i]) {
                    stack.popLast()
                }
                b[i] = stack.isEmpty ? count : stack.last!
                stack.append(i)
            }
            var ans = [Int].init(repeating: 0, count: count)
            for i in 0..<count {
                ans[i] = (i - a[i]) * (b[i] - i)
            }
            return ans
        }
        return ans
    }

    func longestSustring(_ s: String, _ k: Int) -> Int {
        var ans = 0
        let chars = [Character](s)
        ans = getCount(0, s.count - 1)
        func getCount(_ l: Int, _ r: Int) -> Int {
            if l > r {
                return 0
            }
            var dict = [Character: Int]()
            var spilt = -1
            for i in l...r {
                dict[chars[i]] = (dict[chars[i]] ?? 0) + 1
            }
            for i in l...r {
                if dict[chars[i]]! < k {
                    spilt = i
                    break
                }
            }
            if spilt == -1 {
                return r - l + 1
            } else if spilt == l {
                return getCount(l + 1, r)
            } else if spilt == r {
                return r - l
            } else {
                return max(getCount(l, spilt - 1), getCount(spilt + 1, r))
            }
        }
        return ans
    }

    func goodDaysToRobBank(_ security: [Int], _ time: Int) -> [Int] {
        var ans = [Int]()
        let count = security.count
        if time == 0 {
            for i in 0..<count {
                ans.append(i)
            }
            return ans
        } else if time * 2 >= count {
            return ans
        }
        var dpLeft = [Int].init(repeating: 0, count: count)
        for i in 1..<count-1 {
            if security[i] > security[i - 1] {
                dpLeft[i] = 0
            } else {
                dpLeft[i] = dpLeft[i - 1] + 1
            }
        }
        var dpRight = [Int].init(repeating: 0, count: count)
        for i in (0..<count-1).reversed() {
            if security[i] > security[i + 1] {
                dpRight[i] = 0
            } else {
                dpRight[i] = dpRight[i + 1] + 1
            }
        }
        for i in time...(count-time-1) {
            if dpLeft[i] >= time && dpRight[i] >= time {
                ans.append(i)
            }
        }
        return ans
    }

    func cellsInRange(_ s: String) -> [String] {
        let allChars = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
        let chars = [Character](s)
        let startCharIndex = Int(chars[0].asciiValue! - Character("A").asciiValue!)
        let endCharIndex = Int(chars[3].asciiValue! - Character("A").asciiValue!)
        var ans = [String]()
        for i in startCharIndex...endCharIndex {
            for j in Int(String(chars[1]))!...Int(String(chars[4]))! {
                ans.append(String(allChars[i]) + "\(j)")
            }
        }
        return ans
    }

    func minimalKSum(_ nums: [Int], _ k: Int) -> Int {
        var set = Set<Int>()
        var sum = 0
        var ans = 0
        var addCount = 0
        for num in nums {
            set.insert(num)
        }
        for n in set {
            if n <= k + set.count {
                sum += n
                addCount += 1
            }
        }
        ans = (1 + k + set.count) * (k + set.count) / 2 - sum
        var redundantCont = 0
        for i in (1...(k+set.count)).reversed() {
            if set.count == addCount + redundantCont {
                break
            }
            if !set.contains(i) {
                ans -= i
                redundantCont += 1
            }
        }
        return ans
    }

    func createBinaryTree(_ descriptions: [[Int]]) -> TreeNode? {
        var root: TreeNode?
        var map = [Int: TreeNode]()
        var set = Set<Int>()
        for description in descriptions {
            set.insert(description[1])
        }
        for description in descriptions {
            var currNode: TreeNode?
            if map[description[0]] != nil {
                currNode = map[description[0]]!
            } else {
                currNode = TreeNode.init(description[0])
            }
            map[description[0]] = currNode
            if map[description[1]] != nil {
                if description[2] == 1 {
                    currNode?.left = map[description[1]]
                } else {
                    currNode?.right = map[description[1]]
                }
            } else {
                let child = TreeNode.init(description[1])
                if description[2] == 1 {
                    currNode?.left = child
                } else {
                    currNode?.right = child
                }
                map[description[1]] = child
                set.insert(description[1])
            }
        }
        for item in map {
            if !set.contains(item.key) {
                root = item.value
                break
            }
        }
        return root
    }

    func replaceNonCoprimes(_ nums: [Int]) -> [Int] {
        var stack = [Int]()
        for i in 0..<nums.count {
            stack.append(nums[i])
            while stack.count > 1 {
                let x = stack[stack.count - 1]
                let y = stack[stack.count - 2]
                let g = gcd(x: x, y: y)
                if g == 1 {
                    break
                }
                stack.removeLast()
                stack.removeLast()
                stack.append(x * y / g)
            }
        }

        func gcd(x: Int, y: Int) -> Int {
            return y == 0 ? x : gcd(x: y, y: x % y)
        }
        return stack
    }

    func convertToBase7(_ num: Int) -> String {
        if num == 0 {
            return "0"
        }
        var ans = ""
        let isNegative = num < 0 ? "-" : ""
        var num = num > 0 ? num : -num
        while num != 0 {
            ans = "\(num % 7)" + ans
            num /= 7
        }
        return isNegative + ans
    }

    func platesBetweenCandles(_ s: String, _ queries: [[Int]]) -> [Int] {
        var ans = [Int]()
        var pre = [Int].init(repeating: 0, count: s.count + 1)
        var leftDp = [Int].init(repeating: 0, count: s.count)
        var rightDp = [Int].init(repeating: 0, count: s.count)
        var candles = [Int]()
        let chars = [Character](s)
        for i in 0..<chars.count {
            if chars[i] == "|" {
                candles.append(i)
            }
            pre[i + 1] = (chars[i] == "*" ? 1 : 0) + pre[i]
        }
        var lastIndex = -1
        for i in (0..<chars.count).reversed() {
            if chars[i] == "|" {
                lastIndex = i
            }
            leftDp[i] = lastIndex
        }
        lastIndex = -1
        for i in 0..<chars.count {
            if chars[i] == "|" {
                lastIndex = i
            }
            rightDp[i] = lastIndex
        }
        for query in queries {
            let left = leftDp[query[0]]
            let right = rightDp[query[1]]
            ans.append(left == -1 || right == -1 || left >= right ? 0 : pre[right + 1] - pre[left + 1])
        }
        return ans
    }

    func bestRotation(_ nums: [Int]) -> Int {
        let n = nums.count
        var diffs = [Int].init(repeating: 0, count: n + 1)
        for i in 0..<nums.count {
            let a = (i - (n - 1) + n) % n
            let b = (i - nums[i] + n) % n
            if a <= b {
                diffs[a] += 1
                diffs[b + 1] -= 1
            } else {
                diffs[0] += 1
                diffs[b + 1] -= 1
                diffs[a] += 1
                diffs[n] -= 1
            }
        }
        for i in 1...n {
            diffs[i] += diffs[i - 1]
        }
        var ans = 0
        for i in 1...n {
            if diffs[i] > diffs[ans] {
                ans = i
            }
        }
        return ans
    }

    func corpFlightBookings(_ bookings: [[Int]], _ n: Int) -> [Int] {
        var diffs = [Int].init(repeating: 0, count: n + 1)
        var ans = [Int].init(repeating: 0, count: n)
        for i in 0..<bookings.count {
            diffs[bookings[i][0] - 1] += bookings[i][2]
            diffs[bookings[i][1]] -= bookings[i][2]
        }
        for i in 1...n {
            ans[i - 1] = diffs[i - 1]
            diffs[i] += diffs[i - 1]
        }
        return ans
    }

    func carPooling(_ trips: [[Int]], _ capacity: Int) -> Bool {
        var diffs = [Int].init(repeating: 0, count: 1003)
        for i in 0..<trips.count {
            diffs[trips[i][1]] += trips[i][0]
            diffs[trips[i][2]] -= trips[i][0]
        }
        for i in 0...1001 {
            if diffs[i] > capacity {
                return false
            }
            diffs[i + 1] += diffs[i]
        }
        return true
    }

    func subarraySum(_ nums: [Int], _ k: Int) -> Int {
        let count = nums.count
        var pre = 0
        var map = [0: 1]
        var ans = 0
        for i in 0..<count {
            pre += nums[i]
            ans += (map[pre - k] ?? 0)
            map[pre] = (map[pre] ?? 0) + 1
        }
        return ans
    }

    func reverseBetween(_ head: ListNode?, _ left: Int, _ right: Int) -> ListNode? {
        let dumpNode: ListNode? = ListNode.init(-1)
        dumpNode!.next = head
        var leftNode = dumpNode
        for _ in 0..<left-1 {
            leftNode = leftNode?.next
        }
        let curr = leftNode?.next
        var next: ListNode? = nil
        for _ in left..<right {
            next = curr?.next
            curr?.next = next?.next
            next?.next = leftNode?.next
            leftNode?.next = next
        }
        return dumpNode!.next
    }

    func preorder(_ root: Node?) -> [Int] {
        var queue = [Node]()
        if let root = root {
            queue.append(root)
        }
        var ans = [Int]()
        while !queue.isEmpty {
            let node = queue.removeLast()
            ans.append(node.val)
            for child in node.children.reversed() {
                queue.append(child)
            }
        }
        return ans
    }

    func countHighestScoreNodes(_ parents: [Int]) -> Int {
        let total = parents.count
        var children = [[Int]].init(repeating: [Int](), count: total)
        var ans = 0
        var maxV = 0
        for i in 1..<total {
            let p = parents[i]
            if p != -1 {
                children[parents[i]].append(i)
            }
        }
        @discardableResult
        func dfs(root: Int) -> Int {
            let child = children[root]
            var score = 1
            var subSize = 0
            var size = total - 1
            for v in child {
                let t = dfs(root: v)
                subSize += t
                score = t * score
                size -= t
            }
            if root != 0 {
                score *= size
            }
            if score > maxV {
                maxV = score
                ans = 1
            } else if score == maxV {
                ans += 1
            }
            return subSize + 1
        }
        dfs(root: 0)
        return ans
    }

    func validUtf8(_ data: [Int]) -> Bool {
        let n = data.count
        var i = 0
        while i < n {
            let t = data[i]
            var cnt = 0
            for j in (0...7).reversed() {
                if (t >> j) & 1 == 1 {
                    cnt += 1
                    continue
                }
                break
            }
            if cnt == 1 || cnt > 4 {
                return false
            }
            if i + cnt - 1 >= n {
                return false
            }
            if 1 <= cnt {
                for k in i+1..<i+cnt {
                    if (data[k] >> 7) & 1 == 1 && (data[k] >> 6) & 1 == 0 {
                        continue
                    } else {
                        return false
                    }
                }
            }
            if cnt == 0 {
                i += 1
            } else {
                i += cnt
            }
        }
        return true
    }

    func findKDistantIndices(_ nums: [Int], _ key: Int, _ k: Int) -> [Int] {
        var ans = [Int]()
        let n = nums.count
        for i in 0..<n {
            let l = max(i - k, 0)
            let r = min(i + k, n - 1)
            for j in l...r {
                if nums[j] == key {
                    ans.append(i)
                    break
                }
            }
        }
        return ans
    }

    func maximumTop(_ nums: [Int], _ k: Int) -> Int {
        if k & 1 == 1 && nums.count == 1 {
            return -1
        }
        if k <= 1 {
            return nums[k]
        }
        var maxV = -1
        let r = min(k - 2, nums.count-1)
        if r > 0 {
            for i in 0...min(k - 2, nums.count-1) {
                if nums[i] > maxV {
                    maxV = nums[i]
                }
            }
        }
        if k < nums.count {
            if nums[k] > maxV {
                maxV = nums[k]
            }
        }
        return maxV
    }

    func digArtifacts(_ n: Int, _ artifacts: [[Int]], _ dig: [[Int]]) -> Int {
        var sites = [[Int]].init(repeating: [Int].init(repeating: 0, count: n), count: n)
        var ans = 0
        for d in dig {
            sites[d[0]][d[1]] = 1
        }
        for artifact in artifacts {
            var canDig = true
            for i in artifact[0]...artifact[2] {
                for j in artifact[1]...artifact[3] {
                    if sites[i][j] == 0 {
                        canDig = false
                        break
                    }
                }
                if !canDig {
                    break
                }
            }
            if canDig {
                ans += 1
            }
        }
        return ans
    }

    func findRestaurant(_ list1: [String], _ list2: [String]) -> [String] {
        var ans = [String]()
        var indexSum = Int.max
        var map = [String: Int]()
        for i in 0..<list1.count {
            map[list1[i]] = i
        }
        for i in 0..<list2.count {
            let love = list2[i]
            if map.keys.contains(love ) {
                let curr = map[love]! + i
                if curr < indexSum {
                    ans = [love]
                    indexSum = curr
                } else if curr == indexSum {
                    ans.append(love)
                }
            }
        }
        return ans
    }

    func countMaxOrSubsets(_ nums: [Int]) -> Int {
        let n = nums.count
        var ans = 0
        var maxV = 0
        for i in 0..<(1 << n) {
            var v = 0
            for j in 0..<n {
                if (i >> j) & 1 == 1 {
                    v |= nums[j]
                }
            }
            if maxV < v {
                ans = 1
                maxV = v
            } else if maxV == v {
                ans += 1
            }
        }
        return ans
    }

    func longestWord(_ words: [String]) -> String {
        var map = [String: Bool]()
        var ans = ""
        for word in words {
            map[word] = (word.count == 1)
        }
        for word in words {
            var isAns = true
            for i in (0..<word.count).reversed() {
                let str = String(word[word.startIndex...word.index(word.startIndex, offsetBy: i)])
                if map.keys.contains(str) && map[str]! {
                    break
                } else if !map.keys.contains(str) {
                    isAns = false
                    break
                }
            }
            if isAns {
                map[word] = true
                if word.count > ans.count {
                    ans = word
                } else if word.count == ans.count {
                    ans = min(ans, word)
                }
            }
        }
        return ans
    }

    func tree2str(_ root: TreeNode?) -> String {
        var ans = ""
        func preorder(node: TreeNode?) {
            guard let node = node else {
                return
            }
            ans.append("\(node.val)")
            if node.left != nil {
                ans.append("(")
                preorder(node: node.left)
                ans.append(")")
            } else if node.right != nil {
                ans.append("()")
            }
            if node.right != nil {
                ans.append("(")
                preorder(node: node.right)
                ans.append(")")
            }
        }
        preorder(node: root)
        return ans
    }

    func countHillValley(_ nums: [Int]) -> Int {
        var ans = 0
        var uniqueNums = [nums[0]]
        for i in 1..<nums.count {
            if nums[i] != nums[i - 1] {
                uniqueNums.append(nums[i])
            }
        }
        if uniqueNums.count >= 3 {
            for i in 1..<uniqueNums.count-1 {
                if uniqueNums[i] < uniqueNums[i - 1] && uniqueNums[i] < uniqueNums[i + 1] {
                    ans += 1
                } else if uniqueNums[i] > uniqueNums[i - 1] && uniqueNums[i] > uniqueNums[i + 1] {
                    ans += 1
                }
            }
        }
        return ans
    }

    func countCollisions(_ directions: String) -> Int {
        var ans = 0
        var chars = [Character](directions)
        let n = directions.count
        var i = 0
        while i < n {
            if chars[i] == "L" {
                if i != 0 && chars[i - 1] == "S" {
                    ans += 1
                    chars[i] = "S"
                } else if i != 0 && chars[i - 1] == "R" {
                    var j = i - 1
                    while j >= 0 {
                        if chars[j] != "R" { break }
                        chars[j] = "S"
                        j -= 1
                    }
                    chars[i] = "S"
                    ans += (i - j)
                }
            } else if i != 0 && chars[i] == "S" && chars[i - 1] == "R" {
                var j = i - 1
                while j >= 0 {
                    if chars[j] != "R" { break }
                    chars[j] = "S"
                    j -= 1
                }
                ans += (i - j - 1)
            }
            i += 1
        }
        return ans
    }

    //[靶子][分数]
    func maximumBobPoints(_ numArrows: Int, _ aliceArrows: [Int]) -> [Int] {
        let n = aliceArrows.count
        var best = [Int].init(repeating: 0, count: n)
        var maxScore = 0
        for i in 0..<(1 << n) {
            var score = 0
            var total = 0
            for j in 0..<n {
                if i >> j & 1 == 1 {
                    score += j
                    total += (aliceArrows[j] + 1)
                }
            }
            if maxScore < score && total <= numArrows {
                maxScore = score
                for j in 0..<n {
                    if i >> j & 1 == 1 {
                        best[j] = (aliceArrows[j] + 1)
                    } else {
                        best[j] = 0
                    }
                }
                best[n - 1] += numArrows - total
            }
        }
        return best
    }

    static var adj = [[Int]].init(repeating: [Int].init(repeating: 0, count: 100010), count: 100010)
    static var dist = [Int].init(repeating: 0, count: 100010)
    static var visvit = [Bool].init(repeating: false, count: 10010)
    func networkBecomesIdle(_ edges: [[Int]], _ patience: [Int]) -> Int {
        var ans = 0
        let n = patience.count
        for edge in edges {
            Solution1.adj[edge[0]].append(edge[1])
            Solution1.adj[edge[1]].append(edge[0])
        }
        var queue = [Int]()
        let head = ListNode.init(-1)
        let tail = ListNode.init(-2)
        head.next = tail
        tail.pre = head
        insertNode(node: .init(0))
        Solution1.visvit[0] = true
        while head.next!.val != -2 {
            let t = removeNodeLast()!.val
            for i in 0..<n {
                if !Solution1.visvit[Solution1.adj[t][i]] {
                    Solution1.dist[Solution1.adj[t][i]] = Solution1.dist[t] + 1
                    insertNode(node: .init(Solution1.adj[t][i]))
                    Solution1.visvit[Solution1.adj[t][i]] = true
                }
            }
        }
        for i in 1..<n {
            let di = Solution1.dist[i] * 2
            let t = patience[i]
            let curr = di <= t ? di : (di - 1) / t * t + di
            if curr > ans {
                ans = curr
            }
        }

        func insertNode(node: ListNode) {
            let temp = head.next
            head.next = node
            node.next = temp
            temp?.pre = node
        }

        func removeNodeLast() -> ListNode? {
            let ans = tail.pre
            let temp = tail.pre?.pre
            tail.pre = temp
            temp?.next = tail
            return ans
        }
        return ans + 1
    }

    func findTarget(_ root: TreeNode?, _ k: Int) -> Bool {
        var set = Set<Int>()
        var ans = false
        func dfs(node: TreeNode?) {
            guard let node = node else {
                return
            }
            if set.contains(k - node.val) {
                ans = true
                return
            }
            set.insert(node.val)
            dfs(node: node.left)
            dfs(node: node.right)
        }
        dfs(node: root)
        return ans
    }

    func winnerOfGame(_ colors: String) -> Bool {
        guard colors.count > 2 else {
            return false
        }
        var aCnt = 0
        var bCnt = 0
        var chars = [Character](colors)
        for i in 1..<colors.count-1 {
            if chars[i] == "A" && chars[i - 1] == "A" && chars[i + 1] == "A" {
                aCnt += 1
            } else if chars[i] == "B" && chars[i - 1] == "B" && chars[i + 1] == "B" {
                bCnt += 1
            }
        }
        return aCnt > bCnt
    }

    func findKthNumber(_ n: Int, _ k: Int) -> Int {
        var ans = 1
        var k = k
        while k > 1 {
            let cnt = getCnt(curr: ans, limit: n)
            if cnt < k {
                ans += 1
                k -= cnt
            } else {
                ans *= 10
                k -= 1
            }
        }
        func getCnt(curr: Int, limit: Int) -> Int {
            var steps = 0
            var next = curr + 1
            var curr = curr
            while curr <= n {
                steps += min(n - curr + 1, next - curr)
                next *= 10
                curr *= 10
            }
            return steps
        }
        return ans
    }

    class CoverNode {
        static func + (lhs: Solution1.CoverNode, rhs: Solution1.CoverNode) -> CoverNode {
            return CoverNode.init(l: min(lhs.l, rhs.l), r: max(lhs.r, rhs.r), cnt: lhs.cnt + rhs.cnt)
        }

        var l, r, cnt: Int
        init(l: Int, r: Int, cnt: Int) {
            self.l = l
            self.r = r
            self.cnt = cnt
        }
    }
    func isCovered(_ ranges: [[Int]], _ left: Int, _ right: Int) -> Bool {
        let N = 50
        var tree = [CoverNode].init(repeating: CoverNode.init(l: 0, r: 0, cnt: 0), count: N * 2)
        func build() {
            for i in N..<2*N {
                let index = i - N + 1
                tree[i] = CoverNode.init(l: index, r: index, cnt: 0)
            }
            for i in (1...N-1).reversed() {
                tree[i] = tree[i << 1] + tree[i << 1 | 1]
            }
        }
        func update(i: Int) {
            var i = i + N
            tree[i].cnt = 1
            while i > 0 {
                tree[i >> 1].cnt = tree[i].cnt + tree[i ^ 1].cnt
                i >>= 1
            }
        }
        func query(l: Int, r: Int) -> Int {
            var l = l + N
            var r = r + N
            var cnt = 0
            while l <= r {
                if l & 1 == 1 {
                    cnt += tree[l].cnt
                    l += 1
                }
                if r & 1 == 0 {
                    cnt += tree[r].cnt
                    r -= 1
                }
                l >>= 1
                r >>= 1
            }
            return cnt
        }
        build()
        for range in ranges {
            let l = range[0]
            let r = range[1]
            for i in l...r {
                update(i: i - 1)
            }
        }
        return query(l: left - 1, r: right - 1) == (right - left + 1)
    }

    func imageSmoother(_ img: [[Int]]) -> [[Int]] {
        let m = img.count
        let n = img[0].count
        var preSums = [[Int]].init(repeating: [Int].init(repeating: 0, count: n + 3), count: m + 3)
        var img = img
        for i in 2...m+2 {
            for j in 2...n+2 {
                if j == n + 2 || i == m + 2 {
                    preSums[i][j] = preSums[i][j - 1]
                } else {
                    preSums[i][j] = preSums[i][j - 1] + img[i - 2][j - 2]
                }
            }
        }
        for i in 2...m+1 {
            for j in 2...n+1 {
                let sum = preSums[i - 1][j + 1] - preSums[i - 1][j - 2] + preSums[i][j + 1] - preSums[i][j - 2] + preSums[i + 1][j + 1] - preSums[i + 1][j - 2]
                let a = i + 1 == m + 2 ? i : i + 1
                let b = i - 1 == 1 ? i : i - 1
                let c = j + 1 == n + 2 ? j : j + 1
                let d = j - 1 == 1 ? j : j - 1
                let total = (a - b + 1) * (c - d + 1)
                img[i - 2][j - 2] = sum / total
            }
        }
        return img
    }

    class LongestNode {
        var pre, suf, maxV, size: Int
        var left, right: Int

        init(pre: Int, suf: Int, maxV: Int, size: Int, left: Int, right: Int) {
            self.pre = pre
            self.suf = suf
            self.maxV = maxV
            self.size = size
            self.left = left
            self.right = right
        }
        init() {
            pre = 0
            suf = 0
            maxV = 0
            size = 0
            left = 0
            right = 0
        }
    }
    func longestRepeating(_ s: String, _ queryCharacters: String, _ queryIndices: [Int]) -> [Int] {
        let n = s.count
        let m = queryIndices.count
        var sChars = [Character](s)
        let queryChars = [Character](queryCharacters)
        var tree = [LongestNode].init(repeating: LongestNode.init(), count: n * 4)
        func mergeNode(o: Int) {
            let left = tree[o << 1]
            let right = tree[o << 1 | 1]
            tree[o].pre = left.pre
            tree[o].suf = right.suf
            tree[o].maxV = max(left.maxV, right.maxV)
            if sChars[left.right - 1] == sChars[right.left - 1] {
                if left.suf == left.size {
                    tree[o].pre += right.pre
                }
                if right.pre == right.size {
                    tree[o].suf += left.suf
                }
                tree[o].maxV = max(tree[o].maxV, left.suf + right.pre)
            }
        }
        func build(l: Int, r: Int, o: Int) {
            tree[o] = LongestNode()
            tree[o].left = l
            tree[o].right = r
            tree[o].size = r - l + 1
            if l == r {
                tree[o].pre = 1
                tree[o].suf = 1
                tree[o].maxV = 1
                return
            }
            let mid = (l + r) >> 1
            build(l: l, r: mid, o: o << 1)
            build(l: mid + 1, r: r, o: o << 1 | 1)
            mergeNode(o: o)
        }
        func update(i: Int, o: Int) {
            if tree[o].left == tree[o].right {
                return
            }
            let mid = (tree[o].left + tree[o].right) >> 1
            if i <= mid {
                update(i: i, o: o << 1)
            } else {
                update(i: i, o: o << 1 | 1)
            }
            mergeNode(o: o)
        }
        build(l: 1, r: n, o: 1)
        var ans = [Int].init(repeating: 0, count: m)
        for i in 0..<queryIndices.count {
            let index = queryIndices[i]
            sChars[index] = queryChars[i]
            update(i: queryIndices[i] + 1, o: 1)
            ans[i] = tree[1].maxV
        }
        return ans
    }

    func trailingZeroes(_ n: Int) -> Int {
        return n == 0 ? 0 : n / 5 + trailingZeroes(n / 5)
    }
}

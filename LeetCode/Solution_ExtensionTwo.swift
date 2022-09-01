//
//  Solution_ExtensionTwo.swift
//  LeetCode
//
//  Created by haodong xu on 2022/8/9.
//

import Foundation
class Solution2 {
    func addTwoNumbers(_ l1: ListNode?, _ l2: ListNode?) -> ListNode? {
        let ans: ListNode? = ListNode.init(-1)
        var curr = ans
        var l1 = l1
        var l2 = l2
        var flag = 0
        while l1 != nil || l2 != nil {
            let v1 = l1?.val ?? 0
            let v2 = l2?.val ?? 0
            let v = v1 + v2 + flag
            curr?.next = ListNode.init(v % 10)
            flag = v / 10
            curr = curr?.next
            if l1 != nil {
                l1 = l1?.next
            }
            if l2 != nil {
                l2 = l2?.next
            }
        }
        if flag != 0 {
            curr?.next = ListNode.init(flag)
        }
        return ans?.next
    }

    func majorityElement(_ nums: [Int]) -> [Int] {
        var ans = [Int]()
        let n = nums.count
        var a = 0, b = 0
        var cntA = 0, cntB = 0
        for num in nums {
            if num == a && cntA != 0 { cntA += 1 }
            else if num == b && cntB != 0 { cntB += 1 }
            else if cntA == 0 {
                cntA = 1
                a = num
            } else if cntB == 0 {
                cntB = 1
                b = num
            } else {
                cntA -= 1
                cntB -= 1
            }
        }
        cntA = 0
        cntB = 0
        for num in nums {
            if num == a { cntA += 1 }
            else if num == b { cntB += 1 }
        }
        if cntA > n / 3 { ans.append(a) }
        if cntB > n / 3 { ans.append(b) }
        return ans
    }

    func missingRolls(_ rolls: [Int], _ mean: Int, _ n: Int) -> [Int] {
        var ans = [Int]()
        var sum = (rolls.count + n) * mean - rolls.reduce(0, { $0 + $1 })
        if sum < n || sum > n * 6 {
            return ans
        }
        var n = n
        while sum > 0 {
            let avg = sum / n
            ans.append(avg)
            sum -= avg
            n -= 1
        }
        return ans
    }

    func findDifference(_ nums1: [Int], _ nums2: [Int]) -> [[Int]] {
        var set = Set<Int>(), common = Set<Int>()
        for num in nums1 {
            set.insert(num)
        }
        for num in nums2 {
            if set.contains(num) {
                common.insert(num)
            }
        }
        var ans1 = Set<Int>(), ans2 = Set<Int>()
        for num in nums1 {
            if !common.contains(num) {
                ans1.insert(num)
            }
        }
        for num in nums2 {
            if !common.contains(num) {
                ans2.insert(num)
            }
        }

        return [ans1.reversed(), ans2.reversed()]
    }

    func minDeletion(_ nums: [Int]) -> Int {
        var ans = 0, currI = 0
        let n = nums.count
        for (i, _) in nums.enumerated() {
            if currI & 1 == 0 && i < n - 1 && nums[i] == nums[i + 1] { // 偶数
                ans += 1
            } else {
                currI += 1
            }
        }
        if (n - ans) & 1 == 1 {
            ans += 1
        }
        return ans
    }

    func kthPalindrome(_ queries: [Int], _ intLength: Int) -> [Int] {
        let n = queries.count
        var ans = [Int].init(repeating: 0, count: n)
        let base = Int(pow(10.0, Double((intLength + 1) / 2 - 1)))
        for (i, query) in queries.enumerated() {
            if base * 9 < query {
                ans[i] = -1
                continue
            }
            var str = "\(base + query - 1)"
            if intLength & 1 == 0 {
                str = str + str[str.startIndex..<str.endIndex].reversed()
            } else {
                str = str + str[str.startIndex..<str.index(before: str.endIndex)].reversed()
            }
            ans[i] = Int(str)!
        }
        return ans
    }

    func maxValueOfCoins(_ piles: [[Int]], _ k: Int) -> Int {
        let n = piles.count
        var dp = [Int].init(repeating: 0, count: k + 1)
        for i in 1...n {
            var arr = [Int].init(repeating: 0, count: piles[i - 1].count + 1)
            for j in 1...piles[i - 1].count {
                arr[j] = arr[j - 1] + piles[i - 1][j - 1]
            }
            for j in (1...k).reversed() {
                for l in 1...j {
                    if l < arr.count {
                        dp[j] = max(dp[j], arr[l] + dp[j - l])
                    }
                }
            }
        }
        return dp[k]
    }

    func threeSum(_ nums: [Int]) -> [[Int]] {
        if nums.count < 3 {
            return []
        }
        let sorted = nums.sorted()
        var ans = [[Int]]()
        let n = nums.count
        for i in 0..<n-2 {
            if i != 0 && sorted[i] == sorted[i - 1] { continue }
            var l = i + 1, r = n - 1
            while l < r {
                if l != i + 1 && sorted[l] == sorted[l - 1] { l += 1 }
                else if r != n - 1 && sorted[r] == sorted[r + 1] { r -= 1 }
                else if sorted[l] + sorted[r] + sorted[i] == 0 {
                    ans.append([sorted[i], sorted[l], sorted[r]])
                    r -= 1
                    l += 1
                }
                else if sorted[l] + sorted[r] + sorted[i] > 0 { r -= 1 }
                else if sorted[l] + sorted[r] + sorted[i] < 0 { l += 1}
            }
        }
        return ans
    }

    func sortColors(_ nums: inout [Int]) {
        var p0 = 0
        var p2 = nums.count - 1
        for i in 0..<p2 {
            while p2 >= 0 && nums[i] == 2 {
                swap(i: p2, j: i)
                p2 -= 1
            }
            if nums[i] == 0 {
                swap(i: p0, j: i)
                p0 += 1
            }
        }
        func swap( i: Int, j: Int) {
            let temp = nums[i]
            nums[i] = nums[j]
            nums[j] = temp
        }
    }

    func hasAlternatingBits(_ n: Int) -> Bool {
        let x = n ^ (n >> 1)
        return x & (x + 1) == 0
    }
    func maxConsecutiveAnswers(_ answerKey: String, _ k: Int) -> Int {
        let answers = [Character](answerKey), n = answers.count
        func getCnt(_ char: Character) -> Int {
            var pre = 0, tail = 0, currK = 0
            var ans = 0
            while tail < n {
                if answers[tail] != char {
                    currK += 1
                }
                while currK > k {
                    if answers[pre] != char {
                        currK -= 1
                    }
                    pre += 1
                }
                ans = max(ans, tail - pre + 1)
                tail += 1
            }
            return ans
        }
        return max(getCnt("T"), getCnt("F"))
    }

    func merge(_ intervals: [[Int]]) -> [[Int]] {
        let sorted = intervals.sorted { item1, item2 in
            return item1[0] < item2[0]
        }
        let n = sorted.count
        var ans = [[Int]](), i = 0
        while i < n {
            var curr = sorted[i], j = i + 1
            while j < n && curr[0] <= sorted[j][0] && curr[1] >= sorted[j][0] {
                curr[1] = max(curr[1], sorted[j][1])
                j += 1
            }
            i = j
            ans.append(curr)
        }
        return ans
    }

    func selfDividingNumbers(_ left: Int, _ right: Int) -> [Int] {
        func isDivideNum(_ num: Int) -> Bool {
            var pre = num
            while pre > 0 {
                let digit = pre % 10
                if digit == 0 || num % digit != 0 {
                    return false
                }
                pre /= 10
            }
            return true
        }
        return (left...right).filter(isDivideNum)
    }

    func canReorderDoubled(_ arr: [Int]) -> Bool {
        var map = [Int: Int]()
        for num in arr {
            map[num] = (map[num] ?? 0) + 1
        }
        let sorted = map.map({ $0.key }).sorted(by: { abs($0) < abs($1) })
        for key in sorted {
            if map[key] == 0 { continue }
            if let cnt2 = map[key * 2], cnt2 >= map[key]! {
                map[key * 2]! -= map[key]!
                map[key] = 0
            } else {
                return false
            }
        }
        return true
    }

    func strongPasswordChecker(_ password: String) -> Int {
        let chars = [Character](password), n = chars.count
        var A = 0, B = 0, C = 0
        for char in chars {
            if char.isLowercase { A = 1 }
            else if char.isNumber { B = 1 }
            else if char.isUppercase { C = 1 }
        }
        let m = A + B + C
        if n < 6 {
            return max(6 - n, 3 - m)
        } else if n <= 20 {
            var tot = 0, i = 0
            while i < n {
                var j = i
                while j < n && chars[i] == chars[j] {
                    j += 1
                }
                let cnt = j - i
                if cnt >= 3 {
                    tot += (cnt / 3)
                }
                i = j
            }
            return max(tot, 3 - m)
        } else {
            var tot = 0, i = 0
            var remain = [0, 0, 0]
            while i < n {
                var j = i
                while j < n && chars[i] == chars[j] { j += 1 }
                let cnt = j - i
                if cnt >= 3 {
                    tot += (cnt / 3)
                    remain[cnt % 3] += 1
                }
                i = j
            }
            let base = n - 20
            var curr = base
            for i in 0..<3 {
                if i == 2 { remain[i] = tot }
                if remain[i] != 0 && curr > 0 {
                    let t = min(remain[i] * (i + 1), curr)
                    curr -= t
                    tot -= (t / (i + 1))
                }
            }
            return base + max(tot, 3 - m)
        }
    }

    func convertTime(_ current: String, _ correct: String) -> Int {
        var ans = 0
        let currArr = current.split(separator: ":"), corrArr = correct.split(separator: ":")
        let currH = Int(currArr[0])!, currMin = Int(currArr[1])!
        let corrH = Int(corrArr[0])!, corrMin = Int(corrArr[1])!
        var diff = (corrH - currH) * 60 + corrMin - currMin
        func countTime(diff: inout Int, time: Int) -> Int {
            var ans = 0
            if diff >= time {
                ans = diff / time
                diff %= time
            }
            return ans
        }
        ans += countTime(diff: &diff, time: 60)
        ans += countTime(diff: &diff, time: 15)
        ans += countTime(diff: &diff, time: 5)
        ans += countTime(diff: &diff, time: 1)
        return ans
    }

    func findWinners(_ matches: [[Int]]) -> [[Int]] {
        var map = [Int: Int]()
        var allWin = [Int]()
        var lostOne = [Int]()
        for match in matches {
            let a = match[0], b = match[1]
            if map[a] == nil {
                map[a] = 0
            }
            map[b] = (map[b] ?? 0) + 1
        }
        for (key, value) in map.sorted(by: { $0.key < $1.key }) {
            if value == 0 {
                allWin.append(key)
            } else if value == 1 {
                lostOne.append(key)
            }
        }
        return [allWin, lostOne]
    }

    func maximumCandies(_ candies: [Int], _ k: Int) -> Int {
        if candies.reduce(0, { $0 + $1 }) < k { return 0 }
        var l = 1, r = candies.max()!
        while l < r {
            let mid = (l + r + 1) >> 1
            if canSeparate(target: mid, k: k) {
                l = mid
            } else {
                r = mid - 1
            }
        }
        func canSeparate(target: Int, k: Int) -> Bool {
            let cnt = candies.reduce(0) { partialResult, candy in
                return partialResult + (candy / target)
            }
            return cnt >= k
        }
        return l
    }

    func nextGreatestLetter(_ letters: [Character], _ target: Character) -> Character {
        if target >= letters.last! {
            return letters.first!
        }
        var l = 0, r = letters.count - 1
        while l < r {
            let mid = (l + r) >> 1
            if letters[mid] > target {
                r = mid
            } else {
                l = mid + 1
            }
        }
        return letters[l]
    }

    func countPrimeSetBits(_ left: Int, _ right: Int) -> Int {
        var ans = 0
        let primes: Set<Int> = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
        for num in left...right {
            if isPrimeSetBits(num: num) { ans += 1 }
        }

        func isPrimeSetBits(num: Int) -> Bool {
            let cnt = num.nonzeroBitCount
            return primes.contains(cnt)
        }
        return ans
    }

    func hammingWeight(_ n: Int) -> Int {
        var cnt = 0
        var n = n
        if n > 0 {
            cnt += (n & 1)
            n >>= 1
        }
        return cnt
    }

    func findMinHeightTrees(_ n: Int, _ edges: [[Int]]) -> [Int] {
        var ans = [Int]()
        var matrix = [[Int]].init(repeating: [Int](), count: n)
        for edge in edges {
            let p1 = edge[0], p2 = edge[1]
            matrix[p1].append(p2)
            matrix[p2].append(p1)
        }
        var parent = [Int].init(repeating: -1, count: n)
        var path = [Int]()
        let x = bfs(curr: 0, parent: &parent)
        var y = bfs(curr: x, parent: &parent)
        parent[x] = -1
        while y != -1 {
            path.append(y)
            y = parent[y]
        }
        if path.count % 2 == 0 {
            ans.append(path[(path.count / 2) - 1])
        }
        ans.append(path[path.count / 2])
        func bfs(curr: Int, parent: inout [Int]) -> Int {
            var queue = [Int]()
            var visvited = [Bool].init(repeating: false, count: n)
            queue.append(curr)
            visvited[curr] = true
            var node = -1
            while !queue.isEmpty {
                let curr = queue.removeFirst()
                node = curr
                for v in matrix[curr] {
                    if !visvited[v] {
                        visvited[v] = true
                        parent[v] = curr
                        queue.append(v)
                    }
                }
            }
            return node
        }
        return ans
    }

    func rotateString(_ s: String, _ goal: String) -> Bool {
        let n = s.count
        let chars = [Character](s), goals = [Character](goal)
        for i in 0..<n {
            if chars[i] == goals[0] {
                let t1 = s.index(s.startIndex, offsetBy: i)
                if s[t1..<s.endIndex] + s[s.startIndex..<t1] == goal {
                    return true
                }
            }
        }
        return false
    }

    func levelOrder(_ root: Node?) -> [[Int]] {
        var ans = [[Int]]()
        guard let root = root else {
            return ans
        }
        var queue: [Node] = [root]
        while !queue.isEmpty {
            var values = [Int]()
            let cnt = queue.count
            for _ in 0..<cnt {
                let curr = queue.removeFirst()
                values.append(curr.val)
                for i in 0..<curr.children.count {
                    queue.append(curr.children[i])
                }
            }
            ans.append(values)
        }
        return ans
    }

    func isEvenOddTree(_ root: TreeNode?) -> Bool {
        guard let root = root else {
            return false
        }
        var queue: [TreeNode] = [root]
        var level = 0
        while !queue.isEmpty {
            let isUp = level & 1 == 0
            var currV = isUp ? Int.min: Int.max
            let cnt = queue.count
            for _ in 0..<cnt {
                let curr = queue.removeFirst()
                if isUp && (curr.val <= currV || curr.val & 1 == 0) {
                    return false
                } else if !isUp && (curr.val >= currV || curr.val & 1 == 1) {
                    return false
                }
                currV = curr.val
                if let left = curr.left {
                    queue.append(left)
                }
                if let right = curr.right {
                    queue.append(right)
                }
            }
            level += 1
        }
        return true
    }

    func reachingPoints(_ sx: Int, _ sy: Int, _ tx: Int, _ ty: Int) -> Bool {
        var tx = tx, ty = ty
        while tx > sx && ty > sy {
            if tx > ty {
                tx %= ty
            } else {
                ty %= tx
            }
        }
        if tx < sx || ty < sy {
            return false
        }
        if sx == tx && sy == ty {
            return true
        } else if sx == tx {
            return ty % tx == tx
        } else {
            return tx % sy == ty
        }
    }

    static let letters = [".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-","..-","...-",".--","-..-","-.--","--.."]
    static let aAsciiValue = Character.init("a").asciiValue!
    func uniqueMorseRepresentations(_ words: [String]) -> Int {

        var set = Set<String>()
        for word in words {
            var str = ""
            for c in word {
                str += Solution.letters[Int(c.asciiValue! - Solution.aAsciiValue)]
            }
            set.insert(str)
        }
        return set.count
    }

    func largestInteger(_ num: Int) -> Int {
        var odds = [Int](), evens = [Int]()
        var arrs = [Int](), num = num
        while num > 0 {
            let curr = num % 10
            num /= 10
            arrs.append(curr)
            if curr & 1 == 0 {
                evens.append(curr)
            } else {
                odds.append(curr)
            }
        }
        odds.sort(by: { $0 > $1 })
        evens.sort(by: { $0 > $1 })
        var i = 0, j = 0
        var ans = ""
        for k in (0..<arrs.count).reversed() {
            if arrs[k] & 1 == 0 {
                ans += "\(evens[i])"
                i += 1
            } else {
                ans += "\(odds[j])"
                j += 1
            }
        }
        return Int(ans)!
    }

    func minimizeResult(_ expression: String) -> String {
        var ans = ""
        var minV = Int.max
        let nums = expression.split(separator: "+")
        func dfs(l: Int, r: Int) {
            let nums1Index = nums[0].index(nums[0].startIndex, offsetBy: l)
            let x = Int(nums[0][nums[0].startIndex..<nums1Index]) ?? 1
            let y = Int(nums[0][nums1Index..<nums[0].endIndex]) ?? 1
            let nums2Index = nums[1].index(nums[1].startIndex, offsetBy: r)
            let z = Int(nums[1][nums[1].startIndex..<nums2Index]) ?? 1
            let k = Int(nums[1][nums2Index..<nums[1].endIndex]) ?? 1
            let v = x * y * k + x * k * z
            if v < minV {
                minV = v
                if l == 0 && r == nums[1].count {
                    ans = "(\(y)+\(z))"
                } else if l == 0 {
                    ans = "(\(y)+\(z))\(k)"
                } else if r == nums[1].count {
                    ans = "\(x)(\(y)+\(z))"
                } else {
                    ans = "\(x)(\(y)+\(z))\(k)"
                }
            }
            if l < nums[0].count - 1 {
                dfs(l: l + 1, r: r)
            }
            if r < nums[1].count {
                dfs(l: l, r: r + 1)
            }
        }
        dfs(l: 0, r: 1)
        return ans
    }

    //    func maximumProduct(_ nums: [Int], _ k: Int) -> Int {
    //
    //    }
    //    5 -
    //    4 - 6
    //    3 - 3
    //    2 - 1
    func countNumbersWithUniqueDigits(_ n: Int) -> Int {
        if n == 0 { return 1 }
        else if n == 1 { return 10 }
        var diff = 9
        var diffCnt = 10
        for i in 2...n {
            diff *= (11 - i)
            diffCnt += diff
        }
        return diffCnt
    }

    func numberOfLines(_ widths: [Int], _ s: String) -> [Int] {
        var line = 1
        var currWidth = 0
        for c in s {
            let cWidth = widths[Int(c.asciiValue! - Character("a").asciiValue!)]
            if currWidth + cWidth <= 100 {
                currWidth += cWidth
            } else {
                line += 1
                currWidth = cWidth
            }
        }
        return [line, currWidth]
    }

    func maximumWealth(_ accounts: [[Int]]) -> Int {
        return accounts.reduce(0) { partialResult, account in
            return max(partialResult, account.reduce(0, { $0 + $1 }))
        }
    }

    func deserialize(_ s: String) -> NestedInteger {
        if let value = Int(s) {
            return NestedInteger.init(value)
        }
        var stack = [NestedInteger]()
        var isNegative = false
        var num = 0
        for (i, char) in s.enumerated() {
            if char == "[" { stack.append(NestedInteger.init(0)) }
            else if char == "-" { isNegative = true }
            else if char.isNumber { num = num * 10 + char.wholeNumberValue! }
            else if char == "," || char == "]" {
                if s[s.index(s.startIndex, offsetBy: i - 1)].isNumber {
                    if isNegative { num *= -1 }
                    let new = NestedInteger(num)
                    stack.last?.add(elem: new)
                }
                num = 0
                isNegative = false
                if char == "]" && stack.count > 1 {
                    let curr = stack.removeLast()
                    stack.last?.add(elem: curr)
                }
            }
        }
        return stack.removeLast()
    }

    func maximumBeauty(_ flowers: [Int], _ newFlowers: Int, _ target: Int, _ full: Int, _ partial: Int) -> Int {
        let sorted = flowers.sorted()
        let n = sorted.count
        if sorted[0] >= target { return full * n }
        var ans = 0
        var preSum = [Int].init(repeating: 0, count: n + 1)
        var newFlowers = newFlowers
        for i in 0..<n {
            preSum[i + 1] = preSum[i] + sorted[i]
        }
        for i in (0..<n).reversed() {
            if sorted[i] >= target { continue }
            var l = sorted[0], r = target - 1
            while l < r {
                let mid = (r + l + 1) >> 1 // 找到第一个大于等于
                if check(mid, i) { l = mid }
                else { r = mid - 1 }
            }
            ans = max(ans, (n - i - 1) * full + l * partial)
            newFlowers -= (target - sorted[i])
            if newFlowers < 0 { break }
        }
        func check(_ minV: Int, _ idx: Int) -> Bool {
            var l = 0, r = idx
            while l < r {
                let mid = (l + r + 1) >> 1
                if sorted[mid] <= minV { l = mid }
                else { r = mid - 1 }
            }
            let diff = minV * (l + 1) - preSum[l + 1]
            return newFlowers >= diff
        }
        if newFlowers > 0 {
            ans = max(ans, full * n)
        }
        return ans
    }

    func largestPalindrome(_ n: Int) -> Int {
        if n == 1 {
            return 9
        }
        let preV = Int(pow(10.0, Double(n))) - 1
        for i in (0...preV).reversed() {
            var num = i, t = i
            while t > 0 {
                num = 10 * num + t % 10
                t /= 10
            }
            var j = preV
            while j * j >= num {
                if num % j == 0 {
                    return num
                }
                j -= 1
            }
        }
        return -1
    }

    func giveGem(_ gem: [Int], _ operations: [[Int]]) -> Int {
        var gem = gem
        for operation in operations {
            let curr = gem[operation[0]] >> 1
            gem[operation[0]] -= curr
            gem[operation[1]] += curr
        }
        return gem.max()! - gem.min()!
    }

    func perfectMenu(_ materials: [Int], _ cookbooks: [[Int]], _ attribute: [[Int]], _ limit: Int) -> Int {
        var ans = 0
        let n = cookbooks.count
        func dfs(currIndex: Int, currentMaterials: [Int], currFull: Int, currDelicious: Int) {
            if currIndex >= n {
                if currFull >= limit { ans = max(ans, currDelicious) }
                return
            }
            dfs(currIndex: currIndex + 1, currentMaterials: currentMaterials, currFull: currFull, currDelicious: currDelicious)
            let currentCookbooks = cookbooks[currIndex]
            if currentCookbooks[0] <= currentMaterials[0] &&
                currentCookbooks[1] <= currentMaterials[1] &&
                currentCookbooks[2] <= currentMaterials[2] &&
                currentCookbooks[3] <= currentMaterials[3] &&
                currentCookbooks[4] <= currentMaterials[4] {
                var currentMaterials = currentMaterials
                currentMaterials[0] -= currentCookbooks[0]
                currentMaterials[1] -= currentCookbooks[1]
                currentMaterials[2] -= currentCookbooks[2]
                currentMaterials[3] -= currentCookbooks[3]
                currentMaterials[4] -= currentCookbooks[4]
                dfs(currIndex: currIndex + 1, currentMaterials: currentMaterials, currFull: currFull + attribute[currIndex][1], currDelicious: currDelicious + attribute[currIndex][0])
            }
        }
        dfs(currIndex: 0, currentMaterials: materials, currFull: 0, currDelicious: 0)
        return ans
    }

    func getNumber(_ root: TreeNode?, _ ops: [[Int]]) -> Int {
        guard let root = root else {
            return 0
        }
        var arr = [Int]()
        func dfs(node: TreeNode) {
            if let left = node.left { dfs(node: left) }
            arr.append(node.val)
            if let right = node.right { dfs(node: right) }
        }
        dfs(node: root)
        var ans = 0
        for num in arr {
            for op in ops.reversed() {
                if (op[1]...op[2]).contains(num) {
                    if op[0] == 1 {
                        ans += 1
                    }
                    break
                }
            }
        }
        return ans
    }

    //    func defendSpaceCity(_ time: [Int], _ position: [Int]) -> Int {
    //        var
    //    }

    func mostCommonWord(_ paragraph: String, _ banned: [String]) -> String {
        var bannedSet = Set<String>()
        for word in banned {
            bannedSet.insert(word)
        }
        var currWord = ""
        let separators: [Character] = ["!", "?", "'", ",", ";", ".", " "]
        var map = [String: Int]()
        for (_, char) in paragraph.enumerated() {
            if separators.contains(char) {
                if currWord != "" && !bannedSet.contains(currWord) { map[currWord] = (map[currWord] ?? 0) + 1 }
                currWord = ""
            }
            else { currWord += "\(char.lowercased())" }
        }
        if currWord != "" && !bannedSet.contains(currWord) { map[currWord] = (map[currWord] ?? 0) + 1 }
        return map.max(by: { return $0.value < $1.value })?.key ?? ""
    }

    func digitSum(_ s: String, _ k: Int) -> String {
        var s = s
        while s.count > k {
            let strs = s.map { char in return String(char) }
            var newStrs = [String]()
            var curr = 0
            for (i, char) in strs.enumerated() {
                curr += Int(String(char))!
                if (i + 1) % k == 0 {
                    newStrs.append(String(curr))
                    curr = 0
                } else if i == strs.count - 1 {
                    newStrs.append(String(curr))
                }
            }
            s = newStrs.reduce("", { $0 + $1 })
        }
        return "\(s)"
    }

    func minimumRounds(_ tasks: [Int]) -> Int {
        var map = [Int: Int](), ans = 0
        for task in tasks {
            map[task] = (map[task] ?? 0) + 1
        }
        for item in map {
            let value = item.value
            if value == 1 { return -1 }
            ans += ((value + 2) / 3)
        }
        return ans
    }

    func maxTrailingZeros(_ grid: [[Int]]) -> Int {
        let n = grid.count, m = grid[0].count
        var preHFactorsNum = [[[Int]]].init(repeating: [[Int]].init(repeating: [0, 0, 0], count: m + 1), count: n + 1)
        var preVFactorsNum = [[[Int]]].init(repeating: [[Int]].init(repeating: [0, 0, 0], count: m + 1), count: n + 1)
        var ans = 0
        for i in 1...n {
            for j in 1...m {
                let num = grid[i - 1][j - 1]
                let two = findCommonFactorCount(num: num, factor: 2)
                let five = findCommonFactorCount(num: num, factor: 5)
                preHFactorsNum[i][j][0] = preHFactorsNum[i][j - 1][0] + two
                preHFactorsNum[i][j][1] = preHFactorsNum[i][j - 1][1] + five
                preVFactorsNum[i][j][0] = preVFactorsNum[i - 1][j][0] + two
                preVFactorsNum[i][j][1] = preVFactorsNum[i - 1][j][1] + five
            }
        }
        for i in 1...n {
            for j in 1...m {
                let left2 = preHFactorsNum[i][j][0], left5 = preHFactorsNum[i][j][1]
                let right2 = preHFactorsNum[i][m][0] - preHFactorsNum[i][j - 1][0], right5 = preHFactorsNum[i][m][1] - preHFactorsNum[i][j - 1][1]
                let top2 = preVFactorsNum[i - 1][j][0], top5 = preVFactorsNum[i - 1][j][1]
                let bottom2 = preVFactorsNum[n][j][0] - preVFactorsNum[i][j][0], bottom5 = preVFactorsNum[n][j][1] - preVFactorsNum[i][j][1]
                let cnt1 = min(left2 + top2, left5 + top5)
                let cnt2 = min(left2 + bottom2, left5 + bottom5)
                let cnt3 = min(right2 + top2, right5 + top5)
                let cnt4 = min(right2 + bottom2, right5 + bottom5)
                let maxV = [cnt1, cnt2, cnt3, cnt4].max()!
                if maxV > ans {
                    ans = maxV
                }
            }
        }
        func findCommonFactorCount(num: Int, factor: Int) -> Int {
            var num = num, ans = 0
            while num > 0 && num % factor == 0 {
                ans += 1
                num /= factor
            }
            return ans
        }
        return ans
    }

    func longestPath(_ parent: [Int], _ s: String) -> Int {
        let n = parent.count, chars = [Character](s)
        var children = [[Int]].init(repeating: [Int](), count: n)
        var ans = 0
        for i in 1..<n {
            children[parent[i]].append(i)
        }
        @discardableResult
        func dfs(curr: Int) -> Int {
            var maxLength = 1
            for child in children[curr] {
                let len = dfs(curr: child)
                if chars[curr] != chars[child] {
                    ans = max(ans, maxLength + len)
                    maxLength = max(maxLength, len)
                }
            }
            return maxLength
        }
        dfs(curr: 0)
        return ans + 1
    }

    func lexicalOrder(_ n: Int) -> [Int] {
        var ans = [Int]()
        var num = 1
        for _ in 0..<n {
            ans.append(num)
            if num * 10 <= n {
                num *= 10
            } else {
                while num % 10 == 9 || num + 1 > n {
                    num /= 10
                }
                num += 1
            }
        }
        return ans
    }

    func lengthLongestPath(_ input: String) -> Int {
        let chars = [Character]("\n" + input), n = chars.count
        var ans = 0, index = 0
        var stack = [(String, Int)]()
        while index < n {
            if chars[index] == "\n" {
                index += 1
                var hierarchy = 0, curr = "", isDir = true
                while index < n &&  chars[index] == "\t" {
                    hierarchy += 1
                    index += 1
                }
                while index < n && chars[index] != "\n" {
                    curr += String(chars[index])
                    if chars[index] == "." { isDir = false }
                    index += 1
                }
                while !stack.isEmpty && hierarchy <= stack.last!.1 { stack.removeLast() }
                if stack.isEmpty {
                    stack.append((curr, 0))
                    if !isDir { ans = max(ans, curr.count) }
                } else {
                    let next = stack.last!.0 + "/" + curr
                    stack.append((next, hierarchy))
                    if !isDir { ans = max(ans, next.count) }
                }
            }
        }
        return ans
    }

    func toGoatLatin(_ sentence: String) -> String {
        var words = sentence.split(separator: " ")
        let vowel: Set<String> = ["a", "e", "i", "o", "u"]
        var suf = ""
        for (i, word) in words.enumerated() {
            suf += "a"
            if vowel.contains(word[word.startIndex].lowercased()) {
                words[i] = words[i] + "ma" + suf
            } else {
                words[i] = words[i].suffix(word.count - 1) + words[i].prefix(1) + "ma" + suf
            }
        }
        return words.joined(separator: " ")
    }

    func findKthLargest(_ nums: [Int], _ k: Int) -> Int {
        var nums = nums
        nums.insert(-1, at: 0)
        var heapSize = nums.count - 1
        buildMaxHeap(heapSize)
        if heapSize-k+2 <= heapSize {
            for i in (heapSize-k+2...heapSize).reversed() {
                swap(i: 1, j: i)
                heapSize -= 1
                maxHeapify(1)
            }
        }
        func buildMaxHeap(_ heapSize: Int) {
            if heapSize > 1 {
                for i in (1...heapSize/2).reversed() {
                    maxHeapify(i)
                }
            }
        }
        func maxHeapify(_ i: Int) {
            var i = i
            while 2 * i <= heapSize {
                var j = 2 * i
                if j < heapSize && nums[j + 1] > nums[j] { j += 1 }
                if nums[i] > nums[j] { break }
                swap(i: i, j: j)
                i = j
            }
        }
        func swap(i: Int, j: Int) {
            let temp = nums[i]
            nums[i] = nums[j]
            nums[j] = temp
        }
        return nums[1]
    }

    func maxRotateFunction(_ nums: [Int]) -> Int {
        if nums.count == 1 { return 0 }
        let n = nums.count
        let sum = nums.reduce(0, { return $0 + $1 })
        var f = 0
        for i in 0..<n {
            f += (i * nums[i])
        }
        var ans = f
        for i in (1..<n).reversed() {
            f -= (n - 1) * nums[i]
            f += (sum - nums[i])
            ans = max(f, ans)
        }
        return ans
    }

    func binaryGap(_ n: Int) -> Int {
        var lastIndex = -1, n = n
        var ans = 0, cnt = 0
        while n > 0 {
            cnt += 1
            if n & 1 == 1 {
                if lastIndex == -1 {
                    lastIndex = cnt
                } else {
                    ans = max(ans, cnt - lastIndex)
                    lastIndex = cnt
                }
            }
            n >>= 1
        }
        return ans
    }

    func intersection(_ nums: [[Int]]) -> [Int] {
        var ans = Set<Int>.init(nums[0])
        let n = nums.count
        for i in 1..<n {
            ans = ans.intersection(Set<Int>.init(nums[i]))
        }
        return ans.sorted(by: { $0 < $1 })
    }

    func countLatticePoints(_ circles: [[Int]]) -> Int {
        var set = Set<[Int]>()
        for circle in circles {
            for y in circle[1]-circle[2]...circle[1]+circle[2] {
                for x in circle[0]-circle[2]...circle[0]+circle[2] {
                    if inCircle(x: x, y: y, circle: circle) {
                        set.insert([x, y])
                    }
                }
            }
        }
        func inCircle(x: Int, y: Int, circle: [Int]) -> Bool {
            let xLen = Int(abs(x - circle[0]))
            let yLen = Int(abs(y - circle[1]))
            let r = circle[2]
            return xLen * xLen + yLen * yLen <= r * r
        }
        return set.count
    }

    // 差分思想 map去存储每个节点
    func fullBloomFlowers(_ flowers: [[Int]], _ persons: [Int]) -> [Int] {
        let m = persons.count
        var diff = [Int: Int]()
        var ans = [Int].init(repeating: 0, count: m)
        for flower in flowers {
            diff[flower[0]] = (diff[flower[0]] ?? 0) + 1
            diff[flower[1] + 1] = (diff[flower[1] + 1] ?? 0) - 1
        }
        let sorted = diff.sorted { item1, item2 in return item1.key < item2.key }
        let ids = [Int].init(0..<m).sorted { i, j in
            return persons[i] < persons[j]
        } // 记录persons中以从小到大排序的id顺序
        var i = 0, sum = 0
        for id in ids {
            while i < sorted.count && sorted[i].key <= persons[id] {
                sum += sorted[i].value
                i += 1
            }
            ans[id] = sum
        }
        return ans
    }

    func projectionArea(_ grid: [[Int]]) -> Int {
        var zSum = 0, ySum = 0
        let n = grid.count
        var xSum = [Int].init(repeating: 0, count: n)
        for i in 0..<n {
            let m = grid[i].count
            var yMax = 0
            for j in 0..<m {
                zSum += 1
                yMax = max(yMax, grid[i][j])
                xSum[j] = max(xSum[j], grid[i][j])
            }
            ySum += yMax
        }
        return zSum + ySum + xSum.reduce(0, { $0 + $1 })
    }

    func pacificAtlantic(_ heights: [[Int]]) -> [[Int]] {
        let n = heights.count, m = heights[0].count
        let directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        var ans = [[Int]]()
        var pacific = [[Bool]].init(repeating: [Bool].init(repeating: false, count: m), count: n)
        var atlantic = [[Bool]].init(repeating: [Bool].init(repeating: false, count: m), count: n)
        for i in 0..<m {
            dfs(y: 0, x: i, arr: &pacific)
            dfs(y: n - 1, x: i, arr: &atlantic)
        }
        for i in 0..<n {
            dfs(y: i, x: 0, arr: &pacific)
            dfs(y: i, x: m - 1, arr: &atlantic)
        }
        for i in 0..<n {
            for j in 0..<m {
                if pacific[i][j] && atlantic[i][j] {
                    ans.append([i, j])
                }
            }        }
        func dfs(y: Int, x: Int, arr: inout [[Bool]]) {
            arr[y][x] = true
            for direction in directions {
                let nextY = y + direction[0]
                let nextX = x + direction[1]
                if (nextY >= 0 && nextY < n)
                    && (nextX >= 0 && nextX < m)
                    && heights[nextY][nextX] >= heights[y][x]
                    && !arr[nextY][nextX] {
                    dfs(y: nextY, x: nextX, arr: &arr)
                }
            }
        }
        return ans
    }

    func countRectangles(_ rectangles: [[Int]], _ points: [[Int]]) -> [Int] {
        let n = points.count
        let rectangles = rectangles.sorted(by: { $0[1] > $1[1] })
        let ids = [Int].init(0..<n).sorted(by: { points[$0][1] > points[$1][1] })
        var ans = [Int].init(repeating: 0, count: n)
        var xs = [Int]()
        var cnt = 0
        for id in ids {
            let start = cnt
            while cnt < rectangles.count && rectangles[cnt][1] >= points[id][1] {
                xs.append(rectangles[cnt][0])
                cnt += 1
            }
            if start < cnt { xs.sort() }
            ans[id] = cnt - binarySearch(xs: &xs, x: points[id][0])
        }
        func binarySearch(xs: inout [Int], x: Int) -> Int {
            var l = 0, r = xs.count
            while l < r {
                let mid = (l + r) >> 1
                if xs[mid] < x { l = mid + 1 }
                else { r = mid }
            }
            return l
        }
        return ans
    }

    func sortArrayByParity(_ nums: [Int]) -> [Int] {
        var nums = nums
        var l = 0, r = nums.count - 1
        while l < r {
            while l < r && nums[l] & 1 == 0 { l += 1 }
            while l < r && nums[r] & 1 == 1 { r -= 1 }
            if l < r && l < r {
                swap(i: l, j: r)
                l += 1
                r -= 1
            }
        }
        func swap(i: Int, j: Int) {
            let temp = nums[i]
            nums[i] = nums[j]
            nums[j] = temp
        }
        return nums
    }

    func removeDigit(_ number: String, _ digit: Character) -> String {
        var nums = [Character](number), ans = "0"
        for i in 0..<nums.count {
            if nums[i] == digit {
                var temp = ""
                for j in 0..<nums.count {
                    if j != i {
                        temp += String(nums[j])
                    }
                }
                ans = max(ans, temp)
            }
        }
        return ans
    }

    func minimumCardPickup(_ cards: [Int]) -> Int {
        let n = cards.count
        var tail = 0, dict = [Int: Int](), ans = Int.max
        while tail < n {
            if dict.keys.contains(cards[tail]) {
                ans = min(tail - dict[cards[tail]]! + 1, ans)
                dict[cards[tail]] = tail
            }
            dict[cards[tail]] = tail
            tail += 1
        }
        return ans == Int.max ? -1 : ans
    }

    func countDistinct(_ nums: [Int], _ k: Int, _ p: Int) -> Int {
        let n = nums.count
        var set = Set<[Int]>()
        func find(curr: Int) {
            var currK = 0, last = -1
            for i in curr..<n {
                if nums[i] % p == 0 {
                    currK += 1
                }
                if currK == k + 1 {
                    last = i - 1
                    break
                }
            }
            if last == -1 {
                last = n - 1
            }
            var subArr: [Int] = []
            for i in curr...last {
                subArr.append(nums[i])
                set.insert(subArr)
            }
        }
        for i in 0..<n {
            find(curr: i)
        }
        return set.count
    }

    // sumG 是上一次以[i - 1]结尾的子字符串
    // sumG += i - pos[v] 为以[i]结尾的子字符串
    // ans 统计所有
    func appealSum(_ s: String) -> Int {
        let n = s.count, chars = [Character](s)
        var ans = 0, sumG = 0
        var pos = [Int].init(repeating: -1, count: 26)
        for i in 0..<n {
            let v = Int(chars[i].asciiValue! - Character.init("a").asciiValue!)
            if pos[v] == -1 {
                sumG += i + 1
                ans += sumG
            } else {
                sumG += i - pos[v]
                ans += sumG
            }
            pos[v] = i
        }
        return ans
    }

    func getAllElements(_ root1: TreeNode?, _ root2: TreeNode?) -> [Int] {
        var arr1: [Int] = [], arr2: [Int] = []
        dfs(root: root1, arr: &arr1)
        dfs(root: root2, arr: &arr2)
        let n = arr1.count, m = arr2.count
        var ans = [Int].init(repeating: 0, count: n + m)
        var i = 0, j = 0, curr = 0
        while i < n || j < m {
            let a = i < n ? arr1[i] : Int.max, b = j < m ? arr2[j] : Int.max
            if a <= b {
                ans[curr] = a
                i += 1
            } else {
                ans[curr] = b
                j += 1
            }
            curr += 1
        }
        func dfs(root: TreeNode?, arr: inout [Int]) {
            guard let root = root else { return }
            dfs(root: root.left, arr: &arr)
            arr.append(root.val)
            dfs(root: root.right, arr: &arr)
        }
        return ans
    }

    func reorderLogFiles(_ logs: [String]) -> [String] {
        return logs.sorted { log1, log2 in
            let log1Arr = log1.split(separator: " ")
            let log2Arr = log2.split(separator: " ")
            if log1Arr[1][log1Arr[1].startIndex].isNumber || log2Arr[1][log2Arr[1].startIndex].isNumber {
                return !log1Arr[1][log1Arr[1].startIndex].isNumber
            }
            let subStr1 = log1Arr[1...log1Arr.count-1].reduce("", { $0 + $1 + " " })
            let subStr2 = log2Arr[1...log2Arr.count-1].reduce("", { $0 + $1 + " " })
            if subStr1 == subStr2 {
                return log1Arr[0] < log2Arr[0]
            } else {
                return subStr1 < subStr2
            }
        }
    }

    func findTheWinner(_ n: Int, _ k: Int) -> Int {
        var queue = [Int].init(1...n)
        while queue.count > 1 {
            for i in 1..<k {
                queue.append(queue.removeFirst())
            }
            queue.removeFirst()
        }
        return queue.first!
    }

    func numSubarrayProductLessThanK(_ nums: [Int], _ k: Int) -> Int {
        let n = nums.count
        var ans = 0, j = 0, p = 1
        for i in 0..<n {
            p *= nums[i]
            while p >= k && j <= i {
                p /= nums[j]
                j += 1
            }
            ans += (i - j + 1)
        }
        return ans
    }

    func minMutation(_ start: String, _ end: String, _ bank: [String]) -> Int {
        if start == end { return 0 }
        let changeChars: [Character] = ["A", "C", "G", "T"]
        let set = Set<String>.init(bank)
        var queue: [String] = [start], visvited: Set<String> = [start]
        var cnt = 0
        while !queue.isEmpty {
            for _ in 0..<queue.count {
                let curr = queue.removeFirst()
                if curr == end { return cnt }
                for i in 0..<8 {
                    for changeChar in changeChars {
                        var temp = [Character](curr)
                        temp[i] = changeChar
                        let next = String(temp)
                        if set.contains(next) && !visvited.contains(next) {
                            queue.append(next)
                            visvited.insert(next)
                        }
                    }
                }
            }
            cnt += 1
        }
        return -1
    }

    func reversePairs(_ nums: [Int]) -> Int {
        var allNumbers = Set<Int>()
        for num in nums {
            allNumbers.insert(num)
            allNumbers.insert(num * 2)
        }
        var values = [Int: Int](), idx = 0
        for x in allNumbers.sorted(by: { $0 < $1 }) {
            values[x] = idx
            idx += 1
        }
        var res = 0
        let bit = BIT(n: values.count)
        for (_, num) in nums.enumerated() {
            let left = values[num * 2]!, right = values.count - 1
            res += (bit.query(right + 1) - bit.query(left + 1))
            bit.update(values[num]! + 1, 1)
        }
        return res

        class BIT {
            var tree: [Int]
            let n: Int
            init(n: Int) {
                self.n = n
                tree = [Int].init(repeating: 0, count: n + 1)
            }

            func query(_ i: Int) -> Int {
                var i = i, ans = 0
                while i > 0 {
                    ans += tree[i]
                    i -= lowbit(i)
                }
                return ans
            }

            func update(_ i: Int, _ val: Int) {
                var i = i
                while i <= n {
                    tree[i] += val
                    i += lowbit(i)
                }
            }

            func lowbit(_ x: Int) -> Int {
                return x & -x
            }
        }
    }

    func countRangeSum(_ nums: [Int], _ lower: Int, _ upper: Int) -> Int {
        let n = nums.count
        var preSum = [Int].init(repeating: 0, count: n + 1)
        for i in 1...n {
            preSum[i] = preSum[i - 1] + nums[i - 1]
        }
        var ans = 0

        class BIT {
            var tree: [Int]
            let n: Int
            init(n: Int) { self.n = n; tree = [Int].init(repeating: 0, count: n + 1) }
            func query(_ i: Int) -> Int {
                var i = i, ans = 0
                while i > 0 { ans += tree[i]; i -= lowbit(i) }
                return ans
            }
            func update(_ i: Int, _ val: Int) {
                var i = i
                while i <= n { tree[i] += val; i += lowbit(i) }
            }
            func lowbit(_ x: Int) -> Int { return x & -x }
        }
        return ans
    }

    func largestGoodInteger(_ num: String) -> String {
        let chars = [Character](num)
        let n = num.count
        var ans = ""
        for i in 0...n-3 {
            if chars[i] == chars[i + 1] && chars[i + 1] == chars[i + 2] {
                ans = max(ans, "\(chars[i])\(chars[i])\(chars[i])")
            }
        }
        return ans
    }

    func averageOfSubtree(_ root: TreeNode?) -> Int {
        guard let root = root else {
            return 0
        }
        var ans = 0
        @discardableResult
        func dfs(node: TreeNode) -> [Int] {
            var sum = node.val
            var cnt = 1
            if let left = node.left {
                let leftRes = dfs(node: left)
                sum += leftRes[0]
                cnt += leftRes[1]
            }
            if let right = node.right {
                let rightRes = dfs(node: right)
                sum += rightRes[0]
                cnt += rightRes[1]
            }
            if sum / cnt == node.val {
                ans += 1
            }
            return [sum, cnt]
        }
        dfs(node: root)
        return ans
    }

    func countTexts(_ pressedKeys: String) -> Int {
        let map: [Character: Int] = ["2": 3, "3": 3, "4": 3, "5": 3, "6": 3, "7": 4, "8": 3, "9": 4]
        let n = pressedKeys.count
        var ans = 1
        let chars = [Character](pressedKeys)
        var i = 0
        while i < n {
            var j = i
            var last = 0
            var dp = [1]
            while j < n && chars[i] == chars[j] {
                let cnt = map[chars[i]]!
                var curr = 0
                for k in 1...cnt {
                    if j - i - k + 1 >= 0 {
                        curr += dp[j - i - k + 1]
                    }
                }
                curr %= 1000000007
                dp.append(curr)
                last = curr
                j += 1
            }
            ans = (ans * (last % 1000000007) % 1000000007)
            i = j
        }
        return ans
    }

    func hasValidPath(_ grid: [[Character]]) -> Bool {
        let n = grid.count, m = grid[0].count
        var dp = [[[Bool]]].init(repeating: [[Bool]].init(repeating: [Bool].init(repeating: false, count: m + n), count: m + 1), count: n + 1)
        if grid[0][0] == ")" || grid[n - 1][m - 1] == "(" { return false }
        dp[1][1][1] = true
        for i in 1...n {
            for j in 1...m {
                let t = grid[i - 1][j - 1] == "(" ? 1 : -1
                for k in 0..<m+n {
                    let kk = k - t
                    if kk < 0 || kk >= m + n {
                        continue
                    }
                    dp[i][j][k] = dp[i][j][k] || dp[i][j - 1][kk] || dp[i - 1][j][kk]
                }
            }
        }
        return dp[n][m][0]
    }

    func findDuplicates(_ nums: [Int]) -> [Int] {
        let n = nums.count
        var nums = nums, ans = [Int]()
        for i in 0..<n {
            let x = abs(nums[i])
            if nums[x - 1] > 0 {
                nums[x - 1] = -nums[x - 1]
            } else {
                ans.append(x)
            }
        }
        return ans
    }

    func diStringMatch(_ s: String) -> [Int] {
        let n = s.count
        var ans = [Int].init(repeating: 0, count: n + 1)
        var min = 0, max = n
        for (i, c) in s.enumerated() {
            if c == "I" {
                ans[i] = min
                min += 1
            } else {
                ans[i] = max
                max -= 1
            }
        }
        ans[n] = min
        return ans
    }

    func minDeletionSize(_ strs: [String]) -> Int {
        let charsArr = strs.map { str in return [Character](str) }
        let n = charsArr.count
        let m = charsArr[0].count
        var ans = 0
        for i in 0..<m {
            var isIncrease = true
            for j in 1..<n {
                if charsArr[j][i] < charsArr[j - 1][i] { isIncrease = false }
            }
            ans += (isIncrease ? 0 : 1)
        }
        return ans
    }

    func removeNthFromEnd(_ head: ListNode?, _ n: Int) -> ListNode? {
        let dummy: ListNode? = ListNode.init(-1, head)
        var pre = dummy, next = dummy
        for _ in 0...n { next = next?.next }
        while next != nil {
            next = next?.next
            pre = pre?.next
        }
        pre?.next = pre?.next?.next
        return dummy?.next
    }

    func canJump(_ nums: [Int]) -> Bool {
        let n = nums.count
        var right = 0
        for i in 0..<n {
            if i <= right { right = max(right, nums[i] + i) }
        }
        return right >= n - 1
    }

    func oneEditAway(_ first: String, _ second: String) -> Bool {
        let n = first.count, m = second.count
        if first == second { return true }
        if abs(n - m) >= 2 { return false }
        if n > m { return oneEditAway(second, first) }
        let firstChars: [Character] = [Character](first), secondChars: [Character] = [Character](second)
        var i = 0, j = 0, cnt = 0
        while i < n && j < m && cnt <= 1 {
            if firstChars[i] == secondChars[j] {
                i += 1
                j += 1
            } else {
                if n == m {
                    i += 1
                    j += 1
                    cnt += 1
                } else {
                    j += 1
                    cnt += 1
                }
            }
        }
        return cnt <= 1
    }

    func minStickers(_ stickers: [String], _ target: String) -> Int {
        let n = target.count, m = stickers.count, aAsciiValue = Character.init("a").asciiValue!
        var stickersCnts = [[Int]].init(repeating: [Int].init(repeating: 0, count: 26), count: m)
        for (i, sticker) in stickers.enumerated() {
            for char in sticker {
                let j = Int(char.asciiValue! - aAsciiValue)
                stickersCnts[i][j] += 1
            }
        }
        var momo = [String: Int]()
        func dfs(cnt: Int, left: String) -> Int {
            if left == "" { return 0 }
            if cnt >= n + 1 { return n + 1 }
            let sorted = String(left.sorted())
            if momo[sorted] != nil {
                return momo[sorted]!
            }
            var ans = n + 1
            for i in 0..<m {
                var next = ""
                var currCnts = stickersCnts[i]
                for char in left {
                    let j = Int(char.asciiValue! - aAsciiValue)
                    if currCnts[j] == 0 {
                        next.append(char)
                    } else {
                        currCnts[j] -= 1
                    }
                }
                if next != left {
                    ans = min(ans, dfs(cnt: cnt + 1, left: next) + 1)
                }
            }
            momo[String(left.sorted())] = ans
            return ans
        }
        let ans = dfs(cnt: 0, left: target)
        return ans < n + 1 ? ans : -1
    }

    func largestTriangleArea(_ points: [[Int]]) -> Double {
        let n = points.count
        var ans: Double = 0
        for i in 0..<n-2 {
            for j in i+1..<n-1 {
                for k in j+1..<n {
                    ans = max(ans, triangleArea(point1: points[i], point2: points[j], point3: points[k]))
                }
            }
        }
        func triangleArea(point1: [Int], point2: [Int], point3: [Int]) -> Double {
            return 0.5 * Double(abs((point1[0] * point2[1] + point2[0] * point3[1] + point3[0] * point1[1] - point1[1] * point2[0] - point2[1] * point3[0] - point3[1] * point1[0])))
        }
        return ans
    }

    func removeAnagrams(_ words: [String]) -> [String] {
        var ans: [String] = [words[0]]
        for i in 1..<words.count {
            if words[i].sorted() != ans.last!.sorted() {
                ans.append(words[i])
            }
        }
        return ans
    }

    func maxConsecutive(_ bottom: Int, _ top: Int, _ special: [Int]) -> Int {
        let special = special.sorted()
        var ans = 0
        var curr = bottom
        for i in 0..<special.count {
            ans = max(ans, special[i] - curr)
            curr = special[i] + 1
        }
        ans = max(ans, top - curr + 1)
        return ans
    }

    func largestCombination(_ candidates: [Int]) -> Int {
        if candidates.count == 1 {
            return 1
        }
        var bits = [Int].init(repeating: 0, count: 32)
        for var candidate in candidates {
            var i = 0
            while candidate > 0 {
                if candidate & 1 == 1 {
                    bits[i] += 1
                }
                i += 1
                candidate >>= 1
            }
        }
        return bits.max()!
    }

    func isAlienSorted(_ words: [String], _ order: String) -> Bool {
        var map = [Character: Int]()
        for (i, char) in order.enumerated() {
            map[char] = i
        }
        for i in 1..<words.count {
            if !check(words[i - 1], words[i]) {
                return false
            }
        }
        func check(_ first: String, _ second: String) -> Bool {
            let n = first.count, m = second.count
            let firstChars = [Character](first), secondChars = [Character](second)
            var i = 0, j = 0
            while i < n && j < m {
                if firstChars[i] != secondChars[j] {
                    return map[firstChars[i]]! < map[secondChars[j]]!
                }
                i += 1
                j += 1
            }
            if i < n { return false }
            if j < m { return true }
            return true
        }
        return true
    }

    func kthSmallest(_ matrix: [[Int]], _ k: Int) -> Int {
        let n = matrix.count
        var left = matrix[0][0], right = matrix[n - 1][n - 1]
        while left < right {
            let mid = left + (right - left) >> 1
            if getCnt(mid) < k {
                left = mid + 1
            } else {
                right = mid
            }
        }
        func getCnt(_ mid: Int) -> Int {
            var i = n - 1, j = 0
            var cnt = 0
            while i >= 0 && j < n {
                if matrix[i][j] <= mid {
                    cnt += i + 1
                    j += 1
                } else {
                    i -= 1
                }
            }
            return cnt
        }
        return left
    }

    func searchRange(_ nums: [Int], _ target: Int) -> [Int] {
        guard nums.count != 0 else { return [-1, -1] }
        let n = nums.count
        func binarySearch(lower: Bool) -> Int {
            var l = 0, r = n - 1
            while l < r {
                if lower {
                    let mid = l + (r - l) >> 1
                    if target > nums[mid] { l = mid + 1 }
                    else { r = mid }
                } else {
                    let mid = l + (r - l + 1) >> 1
                    if target < nums[mid] { r = mid - 1 }
                    else { l = mid }
                }
            }
            return nums[l] == target ? l : -1
        }
        return [binarySearch(lower: true), binarySearch(lower: false)]
    }

    func findKthNumber(_ m: Int, _ n: Int, _ k: Int) -> Int {
        var l = 1, r = m * n
        while l < r {
            let mid = l + (r - l) >> 1
            if k <= getCnt(mid) { r = mid }
            else { l = mid + 1 }
        }
        func getCnt(_ mid: Int) -> Int {
            var i = n, j = 1
            var cnt = 0
            while i > 0 && j <= m {
                if mid >= i * j {
                    cnt += i
                    j += 1
                } else {
                    i -= 1
                }
            }
            return cnt
        }
        return l
    }

    func minMoves2(_ nums: [Int]) -> Int {
        let n = nums.count
        var nums = nums
        func findKthBy(l: Int, r: Int, target: Int) -> Int {
            let random = Int.random(in: l...r)
            let kth = quickSort(l: l, r: r, curr: random)
            if kth == target {
                return nums[kth]
            } else if kth < target {
                return findKthBy(l: kth + 1, r: r, target: target)
            } else {
                return findKthBy(l: l, r: kth - 1, target: target)
            }
        }
        func quickSort(l: Int, r: Int, curr: Int) -> Int {
            if l == r {
                return l
            }
            swap(i: curr, j: l)
            var j = l + 1
            for i in l+1...r{
                if nums[i] > nums[l] {
                    swap(i: i, j: j)
                    j += 1
                }
            }
            swap(i: j - 1, j: l)
            return j - 1
        }
        func swap(i: Int, j: Int) {
            let temp = nums[i]
            nums[i] = nums[j]
            nums[j] = temp
        }
        let median = findKthBy(l: 0, r: n - 1, target: n / 2)
        var ans = 0
        for num in nums {
            ans += abs(num - median)
        }
        return ans
    }

    func findRightInterval(_ intervals: [[Int]]) -> [Int] {
        let n = intervals.count
        var ans = [Int].init(repeating: -1, count: n)
        var map = [Int: Int]()
        for i in 0..<n {
            map[intervals[i][0]] = i
        }
        let sortedMap = map.sorted { item1, item2 in
            return item1.key < item2.key
        }
        for i in 0..<n {
            var l = 0, r = n
            while l < r {
                let mid = l + (r - l) >> 1
                if sortedMap[mid].key < intervals[i][1] {
                    l = mid + 1
                } else {
                    r = mid
                }
            }
            ans[i] = l < n ? sortedMap[l].value : -1
        }
        return ans
    }

    func repeatedNTimes(_ nums: [Int]) -> Int {
        for (i, num) in nums.enumerated() {
            if i - 1 >= 0 && nums[i - 1] == num {
                return num
            } else if i - 2 >= 0 && nums[i - 2] == num {
                return num
            } else if i - 3 >= 0 && nums[i - 3] == num {
                return num
            }
        }
        return -1
    }

    //    func canIWin(_ maxChoosableInteger: Int, _ desiredTotal: Int) -> Bool {
    //
    //    }

    func percentageLetter(_ s: String, _ letter: Character) -> Int {
        let n = s.count
        var cnt = 0
        for char in s {
            if letter == char {
                cnt += 1
            }
        }
        return cnt * 100 / n
    }

    func maximumBags(_ capacity: [Int], _ rocks: [Int], _ additionalRocks: Int) -> Int {
        let n = capacity.count
        var remains = [Int].init(repeating: 0, count: n)
        var cnt = 0, additionalRocks = additionalRocks
        for i in 0..<n {
            remains[i] = capacity[i] - rocks[i]
        }
        remains.sort()
        for i in 0..<n {
            if remains[i] == 0 {
                cnt += 1
            } else if additionalRocks >= remains[i] {
                cnt += 1
                additionalRocks -= remains[i]
            } else {
                break
            }
        }
        return cnt
    }

    func minimumLines(_ stockPrices: [[Int]]) -> Int {
        if stockPrices.count == 1 {
            return 0
        }
        let n = stockPrices.count
        let stockPrices = stockPrices.sorted { item1, item2 in
            return item1[0] < item2[0]
        }
        var cnt = 0
        func gcd(_ a: Int, _ b: Int) -> Int {
            return b == 0 ? a : gcd(b, a % b)
        }
        func getK(point1: [Int], point2: [Int]) -> (Int, Int) {
            let a = gcd(point2[1] - point1[1], point2[0] - point1[0])
            return ((point2[1] - point1[1]) / a, (point2[0] - point1[0]) / a)
        }
        var lastK: (Int, Int) = (1, 0)
        for i in 1..<n {
            let currK = getK(point1: stockPrices[i - 1], point2: stockPrices[i])
            if (lastK != currK) {
                cnt += 1
                lastK = currK
            }
        }
        return cnt
    }

    //    func totalStrength(_ strength: [Int]) -> Int {
    //        let n = strength.count
    //        var left = [Int].init(repeating: Int.max, count: n + 1)
    //        var right = [Int].init(repeating: Int.max, count: n + 1)
    //        var sumPre = [Int].init(repeating: 0, count: n + 1)
    //        var cnt = 0
    //        for i in 1...n {
    //            sumPre[i] = sumPre[i - 1] + strength[i - 1]
    //        }
    //        for k in 1...i {
    //            let curr = (sumPre[i] - sumPre[k - 1]) * minArr[k][i]
    //            cnt = (cnt + curr) % 1000000007
    //        }
    //        return cnt
    //    }

    func canIWin(_ maxChoosableInteger: Int, _ desiredTotal: Int) -> Bool {
        if (1...maxChoosableInteger).reduce(0, { $0 + $1 }) < desiredTotal { return false }
        if maxChoosableInteger >= desiredTotal { return true }
        let n = maxChoosableInteger
        var memo: [Int: Bool] = [Int: Bool]()
        func dfs(mask: Int, sum: Int) -> Bool {
            if let res = memo[mask] { return res }
            var canWin: Bool = false
            for i in 0..<n {
                let choice: Bool = mask & (1 << i) > 0
                if choice { continue }
                let currSum = sum + i + 1
                if currSum >= desiredTotal {
                    canWin = true
                    break
                } else {
                    canWin = canWin || !dfs(mask: mask | 1 << i, sum: currSum)
                }
            }
            memo[mask] = canWin
            return canWin
        }
        return dfs(mask: 0, sum: 0)
    }

    func nextGreaterElement(_ nums1: [Int], _ nums2: [Int]) -> [Int] {
        let n = nums1.count, m = nums2.count
        var map = [Int: Int]()
        var ans = [Int].init(repeating: 0, count: n)
        var stack = [Int]()
        for i in 0..<m {
            let num = nums2[i]
            while !stack.isEmpty && stack.last! < num {
                map[stack.last!] = num
                stack.removeLast()
            }
            stack.append(num)
        }
        for (i, num) in nums1.enumerated() {
            ans[i] = map[num] ?? -1
        }
        return ans
    }

    func cutOffTree(_ forest: [[Int]]) -> Int {
        if forest[0][0] == 0 { return -1 }
        let n = forest.count, m = forest[0].count
        var list = [[Int]]()
        for (i, forest) in forest.enumerated() {
            for (j, item) in forest.enumerated() {
                if item != 0 && item != 1 {
                    list.append([item, i, j])
                }
            }
        }
        list.sort(by: { return $0[0] < $1[0] })
        let dirs = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        func bfs(x: Int, y: Int, p: Int, q: Int) -> Int {
            var queue = [[y, x]]
            var step = 0
            var visvited: [[Bool]] = [[Bool]].init(repeating: [Bool].init(repeating: false, count: m), count: n)
            visvited[y][x] = true
            while !queue.isEmpty {
                let cnt = queue.count
                for _ in 0..<cnt {
                    let curr = queue.removeFirst()
                    let currX = curr[1], currY = curr[0]
                    if currX == p && currY == q { return step }
                    for dir in dirs {
                        let nx = dir[1] + currX, ny = dir[0] + currY
                        if nx >= 0 && nx < m && ny >= 0 && ny < n && !visvited[ny][nx] && forest[ny][nx] != 0 {
                            queue.append([ny, nx])
                            visvited[ny][nx] = true
                        }
                    }
                }
                step += 1
            }
            return -1
        }
        var ans = 0
        var x = 0, y = 0
        for item in list {
            let nx = item[2], ny = item[1]
            let step = bfs(x: x, y: y, p: nx, q: ny)
            if step == -1 { return -1 }
            ans += step
            x = nx; y = ny
        }
        return ans
    }

    func findSubstringInWraproundString(_ p: String) -> Int {
        var dp = [Int].init(repeating: 0, count: 26)
        let chars = [Character](p), aAsciiValue = Int(Character("a").asciiValue!)
        var cnt = 0
        for (i, char) in chars.enumerated() {
            if i > 0 && (26 + char.asciiValue! - chars[i - 1].asciiValue!) % 26 == 1 {
                cnt += 1
            } else {
                cnt = 1
            }
            let j = Int(char.asciiValue!) - aAsciiValue
            dp[j] = max(dp[j], cnt)
        }
        return dp.reduce(0, { $0 + $1 })
    }

    func largestRectangleArea(_ heights: [Int]) -> Int {
        let n = heights.count
        var right = [Int].init(repeating: n, count: n)
        var left = [Int].init(repeating: -1, count: n)
        var stack = [Int]()
        for (i, height) in heights.enumerated() {
            while !stack.isEmpty && heights[stack.last!] > height {
                right[stack.last!] = i
                stack.removeLast()
            }
            left[i] = stack.isEmpty ? -1 : stack.last!
            stack.append(i)
        }
        var ans = 0
        for (i, height) in heights.enumerated() {
            let cnt = right[i] - left[i] - 1
            ans = max(ans, height * cnt)
        }
        return ans
    }

    func totalStrength(_ strength: [Int]) -> Int {
        let n = strength.count, mod: Int = Int(1e9 + 7)
        var left = [Int].init(repeating: -1, count: n)
        var right = [Int].init(repeating: n, count: n)
        var stack = [Int]()
        var ans = 0
        for i in 0..<n {
            while !stack.isEmpty && strength[stack.last!] > strength[i] {
                right[stack.last!] = i
                stack.removeLast()
            }
            left[i] = stack.isEmpty ? -1 : stack.last!
            stack.append(i)
        }
        var s = 0
        var ss = [Int].init(repeating: 0, count: n + 2)
        for i in 1...n {
            s += strength[i - 1]
            ss[i + 1] = (ss[i] + s) % mod
        }
        for i in 0..<n {
            let l = left[i] + 1, r = right[i] - 1
            let tot = ((i - l + 1) * (ss[r + 2] - ss[i + 1]) - (r - i + 1) * (ss[i + 1] - ss[l])) % mod
            ans = (ans + (tot * strength[i])) % mod
        }
        return (ans + mod) % mod
    }

    func fallingSquares(_ positions: [[Int]]) -> [Int] {
        let n = positions.count
        var heights = [Int].init(repeating: 0, count: n)
        for (i, position) in positions.enumerated() {
            var currHeight = position[1]
            let l2 = position[0], r2 = position[0] + position[1]
            for i in 0..<i {
                let l1 = positions[i][0], r1 = positions[i][0] + positions[i][1]
                if (l2 >= l1 && l2 < r1) || (r2 <= r1 && r2 > l1) || (l2 <= l1 && r2 >= r1) {
                    currHeight = max(currHeight, heights[i] + position[1])
                }
            }
            heights[i] = currHeight
        }
        for i in 1..<n {
            heights[i] = max(heights[i], heights[i - 1])
        }
        return heights
    }

    func findClosest(_ words: [String], _ word1: String, _ word2: String) -> Int {
        var map = [String: [Int]]()
        for (i, word) in words.enumerated() {
            if map.keys.contains(word) {
                map[word]?.append(i)
            } else {
                map[word] = [i]
            }
        }
        if let arr1 = map[word1], let arr2 = map[word2] {
            let n = arr1.count, m = arr2.count
            var i = 0, j = 0
            var ans = Int.max
            while i < n && j < m {
                ans = min(ans, abs(arr1[i] - arr2[j]))
                if arr1[i] < arr2[j] {
                    i += 1
                } else {
                    j += 1
                }
            }
            return ans
        }
        return 0
    }

    func rearrangeCharacters(_ s: String, _ target: String) -> Int {
        var map = [Character: Int]()
        var tMap = [Character: Int]()
        var ans = Int.max
        for char in s {
            map[char] = (map[char] ?? 0) + 1
        }
        for char in target {
            tMap[char] = (tMap[char] ?? 0) + 1
        }
        for tItem in tMap {
            if map[tItem.key] == nil {
                return 0
            }
            ans = min(ans, map[tItem.key]! / tItem.value)
        }
        return ans
    }

    func discountPrices(_ sentence: String, _ discount: Int) -> String {
        var words = sentence.split(separator: " ")
        for (i, word) in words.enumerated() {
            var chars = [Character](word)
            if chars[0] == "$" {
                chars.removeFirst()
                if let price = Int(String(chars)) {
                    let doublePrice = Double(price * (100 - discount)) * 0.01
                    let strPrice = String(format:"%.2f",doublePrice)
                    words[i] = "$" + strPrice
                }
            }
        }
        return words.joined(separator: " ")
    }

    func totalSteps(_ nums: [Int]) -> Int {
        let n = nums.count
        var cnt = 0, i = 1
        while i < nums.count {
            var j = i
            while j < n && nums[j] < nums[i - 1] {
                j += 1
            }
            if i != j {
                var currCnt = 1
                for k in i+1..<j {
                    if nums[k] >= nums[k - 1] {
                        currCnt += 1
                    } else {
                        cnt = max(cnt, currCnt)
                        currCnt = 1
                    }
                }
                cnt = max(cnt, currCnt)
                i = j
            } else {
                i = i + 1
            }
        }
        return cnt
    }

    func validIPAddress(_ queryIP: String) -> String {
        var letters: Set<Character> = ["a", "b", "c", "d", "e", "f", "A", "B", "C", "D", "E", "F"]
        var ips = queryIP.split(separator: ".", omittingEmptySubsequences: false)
        if ips.count == 4 {
            var isIPV4 = true
            for ip in ips {
                if ip.count != 1 && ip.starts(with: "0") {
                    isIPV4 = false
                    break
                }
                if let v = Int(ip), v >= 0 && v <= 255 { } else {
                    isIPV4 = false
                    break
                }
            }
            return isIPV4 && !queryIP.starts(with: ".") && !queryIP.hasSuffix(".") ? "IPv4" : "Neither"
        }
        ips = queryIP.split(separator: ":", omittingEmptySubsequences: false)
        if ips.count == 8 {
            var isIPV6 = true
            for ip in ips {
                if ip.count < 1 || ip.count > 4 {
                    isIPV6 = false
                    break
                }
                for char in ip {
                    if !(letters.contains(char) || char.isNumber) {
                        isIPV6 = false
                        break
                    }
                }
            }
            return isIPV6 && !queryIP.starts(with: ":") && !queryIP.hasSuffix(":") ? "IPv6" : "Neither"
        }
        return "Neither"
    }

    func sumRootToLeaf(_ root: TreeNode?) -> Int {
        guard let root = root else { return 0 }
        func dfs(curr: TreeNode, sum: Int) -> Int {
            if curr.left == nil && curr.right == nil {
                return sum << 1 + curr.val
            }
            var ans = 0
            if let left = curr.left {
                ans += dfs(curr: left, sum: sum << 1 + curr.val)
            }
            if let right = curr.right {
                ans += dfs(curr: right, sum: sum << 1 + curr.val)
            }
            return ans
        }
        return dfs(curr: root, sum: 0)
    }

    func makesquare(_ matchsticks: [Int]) -> Bool {
        let sum = matchsticks.reduce(0, { $0 + $1 })
        guard sum % 4 == 0 else {
            return false
        }
        let matchsticks = matchsticks.sorted { v1, v2 in
            return v1 > v2
        }
        let side = sum / 4, n = matchsticks.count
        var ans = false
        var sides = [Int].init(repeating: 0, count: 4)
        func dfs(idx: Int, sides: inout [Int]) {
            if idx == n {
                ans = sides.reduce(true, { $0 && ($1 == side) })
                return
            }
            for i in 0..<4 {
                let nextSide = sides[i] + matchsticks[idx]
                if nextSide > side {
                    continue
                }
                sides[i] += matchsticks[idx]
                dfs(idx: idx + 1, sides: &sides)
                sides[i] -= matchsticks[idx]
            }
        }
        dfs(idx: 0, sides: &sides)
        return ans
    }

    func deleteNode(_ root: TreeNode?, _ key: Int) -> TreeNode? {
        guard let root = root else {
            return nil
        }
        if root.val > key {
            root.left = deleteNode(root.left, key)
        } else if root.val < key {
            root.right = deleteNode(root.right, key)
        } else {
            if root.left == nil && root.right == nil {
                return nil
            } else if root.left != nil && root.right == nil {
                return root.left
            } else if root.left == nil && root.right != nil {
                return root.right
            } else {
                var rightLeft = root.right
                while rightLeft?.left != nil {
                    rightLeft = rightLeft?.left
                }
                rightLeft?.left = root.left
                return root.right
            }
        }
        return root
    }

    func numUniqueEmails(_ emails: [String]) -> Int {
        let set = Set<String>.init(emails.map { emails in
            let arr = emails.split(separator: "@")
            var trans = ""
            for char in arr[0] {
                if char == "." {
                    continue
                } else if char == "+" {
                    break
                } else {
                    trans.append(char)
                }
            }
            return trans + "@" + arr[1]
        })
        return set.count
    }

    func mergeKLists(_ lists: [ListNode?]) -> ListNode? {
        guard lists.count > 0 else { return nil }
        var ans: ListNode? = lists[0]
        for i in 1..<lists.count {
            ans = mergeTwoList(first: ans, second: lists[i])
        }
        func mergeTwoList(first: ListNode?, second: ListNode?) -> ListNode? {
            if first == nil { return second }
            if second == nil { return first }
            var p = first, q = second
            let head = ListNode.init(-1)
            var tail: ListNode? = head
            while p != nil && q != nil {
                if p!.val <= q!.val {
                    tail?.next = p
                    p = p?.next
                } else {
                    tail?.next = q
                    q = q?.next
                }
                tail = tail?.next
            }
            if p != nil {
                tail?.next = p
            } else if q != nil {
                tail?.next = q
            }
            return head.next
        }
        return ans
    }

    func minMaxGame(_ nums: [Int]) -> Int {
        var nums = nums
        while nums.count > 1 {
            var t = [Int].init(repeating: 0, count: nums.count >> 1)
            var isMin = true
            for i in 0..<nums.count >> 1 {
                t[i] = isMin ? min(nums[i << 1], nums[i << 1 + 1]) : max(nums[i << 1], nums[i << 1 + 1])
                isMin = !isMin
            }
            nums = t
        }
        return nums[0]
    }

    func partitionArray(_ nums: [Int], _ k: Int) -> Int {
        let nums = nums.sorted()
        var cnt = 1
        var minV = nums[0]
        for i in 1..<nums.count {
            if nums[i] - minV > k {
                minV = nums[i]
                cnt += 1
            }
        }
        return cnt
    }

    func arrayChange(_ nums: [Int], _ operations: [[Int]]) -> [Int] {
        var dict = [Int: Int]()
        for (i, num) in nums.enumerated() {
            dict[num] = i
        }
        for operation in operations {
            let id = dict[operation[0]]
            dict.removeValue(forKey: operation[0])
            dict[operation[1]] = id
        }
        return dict.sorted { item1, item2 in
            return item1.value < item2.value
        }.map { item in
            return item.key
        }
    }

    func minEatingSpeed(_ piles: [Int], _ h: Int) -> Int {
        var l = 1, r = Int(1e9)
        while l < r {
            let mid = l + (r - l) >> 1
            if piles.reduce(0, { $0 + Int(ceil(Double($1) * 1.0 / Double(mid))) }) <= h { r = mid }
            else { l = mid + 1 }
        }
        return l
    }

    func isBoomerang(_ points: [[Int]]) -> Bool {
        let v1 = [points[1][0] - points[0][0], points[1][1] - points[0][1]]
        let v2 = [points[2][0] - points[0][0], points[2][1] - points[0][1]]
        return v1[0] * v2[1] != v1[1] * v2[0]
    }

    func countPalindromicSubsequences(_ s: String) -> Int {
        let mod = 1000000007
        let n = s.count
        let chars = [Character](s)
        var dp = [[[Int]]].init(repeating: [[Int]].init(repeating: [Int].init(repeating: 0, count: n), count: n), count: 4)
        let aAsciiValue = Character.init("a").asciiValue!
        for (i, char) in chars.enumerated() {
            dp[Int(char.asciiValue! - aAsciiValue)][i][i] = 1
        }
        if n >= 2 {
            for len in 2...n {
                for i in 0...(n - len) {
                    let j = i + len - 1
                    for c in 0..<4 {
                        let iV = Int(chars[i].asciiValue! - aAsciiValue), jV = Int(chars[j].asciiValue! - aAsciiValue)
                        if iV == c && jV == c {
                            dp[c][i][j] = ((2 + dp[0][i + 1][j - 1] + dp[1][i + 1][j - 1]) % mod + (dp[2][i + 1][j - 1] + dp[3][i + 1][j - 1]) % mod) % mod
                        } else if iV == c {
                            dp[c][i][j] = dp[c][i][j - 1]
                        } else if jV == c {
                            dp[c][i][j] = dp[c][i + 1][j]
                        } else {
                            dp[c][i][j] = dp[c][i + 1][j - 1]
                        }
                    }
                }
            }
        }
        return dp.reduce(0, {($0 + $1[0][n - 1]) % mod})
    }

    func minFlipsMonoIncr(_ s: String) -> Int {
        let n = s.count
        let chars = [Character](s)
        var preSum = [Int].init(repeating: 0, count: n + 1)
        var ans = Int.max
        for i in 1...n {
            preSum[i] = preSum[i - 1] + (chars[i - 1] == "1" ? 1 : 0)
        }
        for i in 1...n {
            let l = preSum[i - 1], r = (n - i) - (preSum[n] - preSum[i])
            ans = min(ans, l + r)
        }
        return ans
    }

    func findAndReplacePattern(_ words: [String], _ pattern: String) -> [String] {
        var ans = [String]()
        let patternChars = [Character](pattern)
        func isMatch(word: [Character], pattern: [Character]) -> Bool {
            let n = word.count
            var map = [Character: Character]()
            for i in 0..<n {
                let wc = word[i], pc = pattern[i]
                if !map.keys.contains(wc) {
                    map[wc] = pc
                } else if map[wc] != pc {
                    return false
                }
            }
            return true
        }
        for word in words {
            if isMatch(word: [Character](word), pattern: patternChars) && isMatch(word: patternChars, pattern: [Character](word)) {
                ans.append(word)
            }
        }
        return ans
    }

    func calculateTax(_ brackets: [[Int]], _ income: Int) -> Double {
        var pb: Double = 0.0
        var ans: Double = 0.0
        for bracket in brackets {
            if income <= bracket[0] {
                let curr: Double = (Double(income) - pb) * Double(bracket[1]) / 100
                ans += curr
                break
            }
            let curr: Double = (Double(bracket[0]) - pb) * Double(bracket[1]) / 100
            ans += curr
            pb = Double(bracket[0])
        }
        return ans
    }

    func minPathCost(_ grid: [[Int]], _ moveCost: [[Int]]) -> Int {
        let m = grid.count, n = grid[0].count
        var dp = [[Int]].init(repeating: [Int].init(repeating: Int.max, count: n), count: m)
        for i in 0..<n {
            dp[0][i] = grid[0][i]
        }
        for i in 1..<m {
            for j in 0..<n {
                for k in 0..<n {
                    dp[i][j] = min(moveCost[grid[i - 1][k]][j] + grid[i][j] + dp[i - 1][k], dp[i][j])
                }
            }
        }
        return dp[m - 1].min()!
    }

    func distributeCookies(_ cookies: [Int], _ k: Int) -> Int {
        var children = [Int].init(repeating: 0, count: k)
        var ans = Int.max
        let n = cookies.count
        func dfs(i: Int, children: inout [Int]) {
            if i == n {
                ans = min(ans, children.max()!)
                return
            }
            for p in 0..<k {
                children[p] += cookies[i]
                dfs(i: i + 1, children: &children)
                children[p] -= cookies[i]
            }
        }
        dfs(i: 0, children: &children)
        return ans
    }

    func distinctNames(_ ideas: [String]) -> Int {
        var group = [String: Int]()
        let aAsciiValue = Character("a").asciiValue!
        for idea in ideas {
            let n = idea.count
            let suf =  String(idea.suffix(n - 1))
            group[suf] = (group[suf] ?? 0) | (1 << (Int(idea[idea.startIndex].asciiValue!) - Int(aAsciiValue)))
        }
        var cnt = [[Int]].init(repeating: [Int].init(repeating: 0, count: 26), count: 26)
        var ans = 0
        for mask in group.values {
            for i in 0..<26 {
                if mask >> i & 1 == 0 { // 当前后缀无此Char
                    for j in 0..<26 {
                        if mask >> j & 1 > 0 {
                            cnt[i][j] += 1
                        }
                    }
                } else { // == 1 当前后缀有这个Char
                    for j in 0..<26 {
                        if mask >> j & 1 == 0 {
                            ans += cnt[i][j]
                        }
                    }
                }
            }
        }
        return ans * 2
    }

    func heightChecker(_ heights: [Int]) -> Int {
        let sortedHeights = heights.sorted(), n = heights.count
        var ans = 0
        for i in 0..<n {
            if sortedHeights[i] != heights[i] {
                ans += 1
            }
        }
        return ans
    }

    func findDiagonalOrder(_ mat: [[Int]]) -> [Int] {
        let n = mat.count, m = mat[0].count
        var leftBottom = [(Int, Int)](), topRight = [(Int, Int)]()
        for i in 0..<n { leftBottom.append((i, 0)) }
        for i in 1..<m { leftBottom.append((n - 1, i)) }
        for i in 0..<m { topRight.append((0, i)) }
        for i in 1..<n { topRight.append((i, m - 1)) }
        var ans = [Int]()
        var dir = 1
        for i in 0..<n+m-1 {
            if dir == 1 { addBy(posotion: leftBottom[i], dir: dir) }
            else { addBy(posotion: topRight[i], dir: dir) }
            dir *= -1
        }
        func addBy(posotion: (Int, Int), dir: Int) {
            var i = posotion.0, j = posotion.1
            if dir == 1 {
                while i >= 0 && j < m {
                    ans.append(mat[i][j])
                    i -= 1
                    j += 1
                }
            } else {
                while i < n && j >= 0 {
                    ans.append(mat[i][j])
                    i += 1
                    j -= 1
                }
            }
        }
        return ans
    }

    func smallestDistancePair(_ nums: [Int], _ k: Int) -> Int {
        let nums = nums.sorted()
        let n = nums.count
        var l = 0, r: Int = Int(1e6)
        func minCnt(x: Int) -> Int {
            var ans = 0
            for i in 0..<n-1 {
                var l = i, r = n - 1
                while l < r {
                    let mid = (l + r + 1) >> 1
                    if nums[mid] - nums[i] > x {
                        r = mid - 1
                    } else {
                        l = mid
                    }
                }
                ans += (l - i)
            }
            return ans
        }
        while l < r {
            let mid = (l + r) >> 1
            if minCnt(x: mid) >= k {
                r = mid
            } else {
                l = mid + 1
            }
        }
        return l
    }

    func findPairs(_ nums: [Int], _ k: Int) -> Int {
        let nums = nums.sorted(), n = nums.count
        var ans = 0
        for i in 0..<n-1 {
            if i >= 1 && nums[i] == nums[i - 1] {
                continue
            }
            if findX(nums[i] + k, l: i + 1, r: n - 1) {
                ans += 1
            }
        }
        func findX(_ x: Int, l: Int, r: Int) -> Bool {
            var l = l, r = r
            while l < r {
                let mid = (l + r) >> 1
                if nums[mid] == x  {
                    return true
                } else if nums[mid] > x {
                    r = mid - 1
                } else {
                    l = mid + 1
                }
            }
            return nums[l] == x
        }
        return ans
    }

    func duplicateZeros(_ arr: inout [Int]) {
        var i = 0, j = 0
        let n = arr.count
        while j < n {
            if arr[i] == 0 {
                j += 1
            }
            i += 1
            j += 1
        }
        i -= 1
        j -= 1
        while i >= 0 {
            if j < n {
                arr[j] = arr[i]
            }
            if arr[i] == 0 {
                j -= 1
                arr[j] = 0
            }
            i -= 1
            j -= 1
        }
    }

    func insert(_ head: SingalNode?, _ insertVal: Int) -> SingalNode? {
        let node = SingalNode(insertVal)
        if head == nil {
            node.next = node
            return node
        }
        let dump = SingalNode(-1)
        dump.next = head
        var curr = head!.next!
        var maxNode = head!
        while curr !== head {
            if curr.val >= maxNode.val {
                maxNode = curr
            }
            curr = curr.next!
        }
        if insertVal >= maxNode.val || maxNode.next!.val >= insertVal {
            node.next = maxNode.next
            maxNode.next = node
        } else {
            curr = maxNode.next!
            while curr !== maxNode {
                if curr.val <= insertVal && curr.next!.val >= insertVal {
                    node.next = curr.next
                    curr.next = node
                    break
                }
                curr = curr.next!
            }
        }
        return dump.next
    }

    func longestConsecutive(_ nums: [Int]) -> Int {
        var ans = 0
        let set = Set<Int>.init(nums)
        for num in set {
            if !set.contains(num - 1) {
                var i = 1
                while set.contains(num + i) {
                    i += 1
                }
                ans = max(ans, i)
            }
        }
        return ans
    }

    func subsets(_ nums: [Int]) -> [[Int]] {
        let n = nums.count
        var ans: [[Int]] = [[]]
        func dfs(_ arr: [Int], _ i: Int) {
            if i >= n { return }
            for j in i..<n {
                var curr = arr
                curr.append(nums[j])
                ans.append(curr)
                dfs(curr, j + 1)
            }
        }
        dfs([], 0)
        return ans
    }

    func subsetsWithDup(_ nums: [Int]) -> [[Int]] {
        let n = nums.count
        let nums = nums.sorted()
        var ans: [[Int]] = [[]]
        func dfs(_ arr: [Int], _ i: Int) {
            if i >= n { return }
            for j in i..<n {
                if j > i && nums[j - 1] == nums[j] {
                    continue
                }
                var curr = arr
                curr.append(nums[j])
                ans.append(curr)
                dfs(curr, j + 1)
            }
        }
        dfs([], 0)
        return ans
    }

    func findFrequentTreeSum(_ root: TreeNode?) -> [Int] {
        guard let root = root else {
            return []
        }
        var map = [Int: Int]()
        var ans = [Int]()
        @discardableResult
        func dfs(node: TreeNode) -> Int {
            var leftSum = 0, rightSum = 0
            if let left = node.left {
                leftSum = dfs(node: left)
            }
            if let right = node.right {
                rightSum = dfs(node: right)
            }
            let sum = leftSum + rightSum + node.val
            map[sum] = (map[sum] ?? 0) + 1
            return sum
        }
        dfs(node: root)
        let cntMax = map.values.max()!
        map.map { item in
            if item.value == cntMax {
                ans.append(item.key)
            }
        }
        return ans
    }

    func greatestLetter(_ s: String) -> String {
        var ans = Character("1")
        var set = Set<Character>()
        for char in s {
            var relativeChar = Character(char.lowercased())
            if char.isLowercase {
                relativeChar = Character(char.uppercased())
            }
            if set.contains(relativeChar) {
                if ans == Character("1") {
                    ans = Character(char.uppercased())
                } else {
                    ans = max(ans, Character(char.uppercased()))
                }
            }
            set.insert(char)
        }
        return ans == "1" ? "" : String(ans)
    }

    func minimumNumbers(_ num: Int, _ k: Int) -> Int {
        if num == 0 {
            return 0
        }
        for u in 1...10 where u * k <= num {
            if (num - u * k) % 10 == 0 {
                return u
            }
        }
        return -1
    }

    func longestSubsequence(_ s: String, _ k: Int) -> Int {
        let n = s.count
        let chars = [Character](s)
        var ans = 0
        chars.map { char in
            if char == "0" {
                 ans += 1
            }
        }
        var num = 0
        for i in (0...n-1).reversed() {
            if chars[i] == "1" {
                if n - i > 64 {
                    break
                }
                if num + (1 << (n - i - 1)) > k {
                    break
                }
                ans += 1
                num += (1 << (n - i - 1))
            }
        }
        return ans
    }

    func defangIPaddr(_ address: String) -> String {
        return address.replacingOccurrences(of: ".", with: "[.]")
    }

    func findBottomLeftValue(_ root: TreeNode?) -> Int {
        guard let root = root else { return 0 }
        var queue: [(TreeNode, Int)] = [(root, 0)]
        var ans = root.val, height = 0
        while !queue.isEmpty {
            let curr = queue.removeFirst()
            if curr.1 > height {
                ans = curr.0.val
                height = curr.1
            }
            if let left = curr.0.left {
                queue.append((left, curr.1 + 1))
            }
            if let right = curr.0.right {
                queue.append((right, curr.1 + 1))
            }
        }
        return ans
    }

    func findAnagrams(_ s: String, _ p: String) -> [Int] {
        let n = s.count, m = p.count
        if n < m { return [] }
        let s = [Character](s), p = [Character](p)
        var ans = [Int]()
        var counts = [Int].init(repeating: 0, count: 26)
        let aV = Character("a").asciiValue!
        for i in 0..<m {
            counts[Int(s[i].asciiValue! - aV)] -= 1
            counts[Int(p[i].asciiValue! - aV)] += 1
        }
        var diff = counts.reduce(0) { partialResult, cnt in
            partialResult + abs(cnt)
        }
        if diff == 0 {
            ans.append(0)
        }
        for i in 0..<n-m {
            let l = Int(s[i].asciiValue! - aV)
            let r = Int(s[i + m].asciiValue! - aV)
            counts[l] += 1
            counts[r] -= 1
            diff = counts.reduce(0) { partialResult, cnt in
                partialResult + abs(cnt)
            }
            if diff == 0 {
                ans.append(i + 1)
            }
        }
        return ans
    }
    // 前序 根 -> 左 -> 右
    // 中序 左 -> 根 -> 右
    func buildTree(_ preorder: [Int], _ inorder: [Int]) -> TreeNode? {
        let n = preorder.count
        var map = [Int: Int]()
        for i in 0..<n {
            map[inorder[i]] = i
        }
        func build(rootIndex: Int, l: Int, r: Int) -> TreeNode? {
            let node = TreeNode(preorder[rootIndex])
            let mid = map[preorder[rootIndex]]!
            if mid != l {
                node.left = build(rootIndex: rootIndex + 1, l: l, r: mid - 1)
            }
            if mid != r {
                node.right = build(rootIndex: rootIndex + mid - l + 1, l: mid + 1, r: r)
            }
            return node
        }
        let ans = build(rootIndex: 0, l: 0, r: n - 1)
        return ans
    }

    func findSubstring(_ s: String, _ words: [String]) -> [Int] {
        let n = s.count, m = words.count, cnt = words[0].count
        if n < m * cnt { return [] }
        let s = [Character](s)
        var map = [String: Int]()
        var ans = [Int]()
        for word in words {
            map[word] = (map[word] ?? 0) + 1
        }
        for i in 0..<cnt {
            var curMap = [String: Int]()
            var j = i
            while j + cnt <= n {
                var str = ""
                for k in j..<j+cnt {
                    str += String(s[k])
                }
                curMap[str] = (curMap[str] ?? 0) + 1
                if (j - i) / cnt >= m {
                    var pre = ""
                    for k in j-cnt*m..<j-cnt*(m-1) {
                        pre += String(s[k])
                    }
                    curMap[pre]! -= 1
                    if curMap[pre] == 0 {
                        curMap.removeValue(forKey: pre)
                    }
                }
                j += cnt
                if map == curMap {
                    ans.append(j-cnt*m)
                }
            }
        }
        return ans
    }

    func largestValues(_ root: TreeNode?) -> [Int] {
        guard let root = root else { return [] }
        var ans = [Int]()
        var queue: [TreeNode] = [root]
        while !queue.isEmpty {
            let n = queue.count
            var mv = Int.min
            for _ in 0..<n {
                let curr = queue.removeFirst()
                mv = max(curr.val, mv)
                if let left = curr.left {
                    queue.append(left)
                }
                if let right = curr.right {
                    queue.append(right)
                }
            }
            ans.append(mv)
        }
        return ans
    }

    func minCost(_ costs: [[Int]]) -> Int {
        let n = costs.count
        var a = costs[0][0], b = costs[0][1], c = costs[0][2]
        for i in 1..<n {
            let newA = min(b, c) + costs[i][0]
            let newB = min(a, c) + costs[i][1]
            let newC = min(a, b) + costs[i][2]
            a = newA; b = newB; c = newC
        }
        return min(a, b, c)
    }

    func checkXMatrix(_ grid: [[Int]]) -> Bool {
        let n = grid.count
        for i in 0..<n {
            if grid[i][i] == 0 || grid[i][n - i - 1] == 0 {
                return false
            }
        }
        for i in 0..<n {
            for j in 0..<n {
                if grid[i][j] != 0 && (i != j && i + j != n - 1) {
                    return false
                }
            }
        }
        return true
    }

    func countHousePlacements(_ n: Int) -> Int {
        let mod = 1000000007
        var dp = [[Int]].init(repeating: [Int].init(repeating: 0, count: 4), count: n + 1)
        dp[1][0] = 0
        dp[1][1] = 1
        dp[1][2] = 1
        dp[1][3] = 1
        if n >= 2 {
            for i in 1...n {
                dp[i][0] = (((((dp[i - 1][0] + dp[i - 1][1]) % mod) + dp[i - 1][2]) % mod) + dp[i - 1][3]) % mod
                dp[i][1] = ((dp[i - 1][0] + dp[i - 1][2]) % mod) + 1 % mod
                dp[i][2] = ((dp[i - 1][0] + dp[i - 1][1]) % mod) + 1 % mod
                dp[i][3] = (dp[i - 1][0] + 1) % mod
            }
        }
        var ans = 0
        for num in dp[n] {
             ans = (ans + num) % mod
        }
        return ans + 1
    }

    func maximumsSplicedArray(_ nums1: [Int], _ nums2: [Int]) -> Int {
        let n = nums1.count
        func find(_ nums1: [Int], _ nums2: [Int]) -> Int {
            let sum = nums1.reduce(0) { partialResult, num in
                partialResult + num
            }
            var ans = sum
            var p = 0, t = 0
            var tempSum = sum
            while t < n {
                if nums1[t] <= nums2[t] {
                    tempSum += (nums2[t] - nums1[t])
                    t += 1
                } else {
                    if tempSum - sum + nums2[t] - nums1[t] > 0 {
                        tempSum += (nums2[t] - nums1[t])
                        t += 1
                    } else {
                        tempSum += nums2[t] - nums1[t]
                        while p <= t {
                            tempSum += nums1[p] - nums2[p]
                            p += 1
                            if tempSum >= sum {
                                break
                            }
                        }
                        if p > t {
                            t = p
                            tempSum = sum
                        }
                    }
                }
                ans = max(ans, tempSum)
            }
            return ans
        }
        return max(find(nums1, nums2), find(nums2, nums1))
    }

    func minRefuelStops(_ target: Int, _ startFuel: Int, _ stations: [[Int]]) -> Int {
        let n = stations.count
        var dp = [[Int]].init(repeating: [Int].init(repeating: 0, count: n + 1), count: n + 1) // 走到i，加j次油的最大油量
        for i in 0...n {
            dp[i][0] = startFuel
        }
        if n >= 1 {
            for i in 1...n {
                for j in 1...i {
                    if dp[i - 1][j] >= stations[i - 1][0] {
                        dp[i][j] = dp[i - 1][j]
                    }
                    if dp[i - 1][j - 1] >= stations[i - 1][0] {
                        dp[i][j] = max(dp[i - 1][j - 1] + stations[i - 1][1], dp[i][j])
                    }
                }
            }
        }
        for i in 0...n {
            if dp[n][i] >= target {
                return i
            }
        }
        return -1
    }

    func maxSubArray(_ nums: [Int]) -> Int {
        let n = nums.count
        var dp = [Int].init(repeating: 0, count: n)
        var ans = nums[0]
        dp[0] = nums[0]
        for i in 1..<n {
            dp[i] = max(dp[n - 1] + nums[i], nums[i])
            ans = max(ans, dp[i])
        }
        return ans
    }

    func decodeMessage(_ key: String, _ message: String) -> String {
        var values = [Character: Character]()
        let letters: [Character] = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l","m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
        var i = 0
        for char in key {
            if char.isLetter && !values.keys.contains(char) {
                values[char] = letters[i]
                i += 1
            }
        }
        var ans = [Character]()
        for char in message {
            if char.isLetter {
                ans.append(values[char]!)
            } else {
                ans.append(char)
            }
        }
        return String(ans)
    }

    func spiralMatrix(_ m: Int, _ n: Int, _ head: ListNode?) -> [[Int]] {
        var ans = [[Int]].init(repeating: [Int].init(repeating: -1, count: n), count: m)
        var i = 0, j = 0
        var head = head
        var currDir = 0
        // left、 bottom、right、top
        let dirs = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        while head != nil {
            ans[i][j] = head!.val
            head = head?.next
            if currDir == 0 && i == 0 && j == n - 1 {
                currDir = 1
                i += dirs[currDir][0]
                j += dirs[currDir][1]
                continue
            } else if currDir == 1 && i == m - 1 && j == n - 1 {
                currDir = 2
                i += dirs[currDir][0]
                j += dirs[currDir][1]
                continue
            } else if currDir == 2 && i == m - 1 && j == 0 {
                currDir = 3
                i += dirs[currDir][0]
                j += dirs[currDir][1]
                continue
            }
            if ans[i + dirs[currDir][0]][j + dirs[currDir][1]] != -1 {
                currDir = ((currDir + 1) % 4)
            }
            i += dirs[currDir][0]
            j += dirs[currDir][1]
        }
        return ans
    }

    func peopleAwareOfSecret(_ n: Int, _ delay: Int, _ forget: Int) -> Int {
        let mod = Int(1e9 + 7)
        var diff = [Int].init(repeating: 0, count: n + forget + 1)
        diff[1] = 1
        for i in 1...n {
            for j in i+delay..<i+forget {
                diff[j] = (diff[j] + diff[i]) % mod
            }
        }
        var ans = 0
        for i in n-forget+1...n {
            ans = ((diff[i] + ans) % mod)
        }
        return ans
    }

    func nextGreaterElement(_ n: Int) -> Int {
        var nums = [Int]()
        var nv = n
        while nv != 0 {
            nums.append(nv % 10)
            nv /= 10
        }
        let cnt = nums.count
        var set = [nums[0]]
        for i in 1..<cnt {
            if nums[i] < nums[i - 1] {
                set.append(nums[i])
                var curr = set.count - 2
                for j in 0..<set.count {
                    if set[j] > nums[i] {
                        if set[curr] > set[j] {
                            curr = j
                        }
                    }
                }
                nums[i] = set[curr]
                set.remove(at: curr)
                nums.removeFirst(i)
                nums.reverse()
                nums.append(contentsOf: set.sorted())
                break
            } else {
                set.append(nums[i])
            }
        }
        let ans = nums.reduce(0, { $0 * 10 + $1 })
        if ans > Int32.max {
            return -1
        }
        return ans > n ? ans : -1
    }

    func minimumAbsDifference(_ arr: [Int]) -> [[Int]] {
        let arr = arr.sorted()
        let n = arr.count
        var minDiff = Int.max
        var ans = [[Int]]()
        for i in 1..<n {
            minDiff = min(arr[i] - arr[i - 1], minDiff)
        }
        for i in 1..<n where minDiff == arr[i] - arr[i - 1] {
            ans.append([arr[i - 1], arr[i]])
        }
        return ans
    }

    func combinationSum(_ candidates: [Int], _ target: Int) -> [[Int]] {
        let n = candidates.count
        var ans = [[Int]]()
        func dfs(sum: Int, arr: inout [Int], index: Int) {
            if index >= n || sum > target { return }
            if sum == target { ans.append(arr) }
            for i in index..<n {
                let v = candidates[i]
                arr.append(v)
                dfs(sum: sum + v, arr: &arr, index: i)
                arr.removeLast()
            }
        }
        var arr = [Int]()
        dfs(sum: 0, arr: &arr, index: 0)
        return ans
    }

    func rob(_ root: TreeNode?) -> Int {
        func dfs(curr: TreeNode?) -> (Int, Int) {
            guard let curr = curr else { return (0, 0) }
            let left = dfs(curr: curr.left)
            let right = dfs(curr: curr.right)
            return (curr.val + left.1 + right.1, max(left.0, left.1) + max(right.0, right.1))
        }
        return max(dfs(curr: root).0, dfs(curr: root).1)
    }

    func maxProfit(_ prices: [Int]) -> Int {
        let n = prices.count
        var dp = [(Int, Int, Int)].init(repeating: (0, 0, 0), count: n)
        dp[0].0 = -prices[0]
        for i in 1..<n {
            dp[i].0 = max(dp[i - 1].0, dp[i - 1].2 - prices[i])
            dp[i].1 = dp[i - 1].0 + prices[i]
            dp[i].2 = max(dp[i - 1].1, dp[i - 1].2)
        }
        return max(dp[n - 1].1, dp[n - 1].2)
    }

    func replaceWords(_ dictionary: [String], _ sentence: String) -> String {
        let set = Set<String>(dictionary)
        var words = sentence.split(separator: " ").map({ return String($0) })
        for (i, word) in words.enumerated() {
            let n = word.count
            for j in 1...n where set.contains(String(word.prefix(j))) {
                break
            }
        }
        return words.joined(separator: " ")
    }

    func minCostToMoveChips(_ position: [Int]) -> Int {
        var oddCnt = 0, evenCnt = 0
        for v in position {
            if v & 1 == 0 { evenCnt += 1 }
            else { oddCnt += 1 }
        }
        return min(oddCnt, evenCnt)
    }

    func lenLongestFibSubseq(_ arr: [Int]) -> Int {
        var ans = 0, map = [Int: Int]()
        let n = arr.count
        for i in 0..<n {
            map[arr[i]] = i
        }
        var dp = [[Int]].init(repeating: [Int].init(repeating: 0, count: n), count: n)
        for i in 1..<n {
            for j in 0...i-1 {
                if let k = map[arr[i] - arr[j]], j > k && k < i {
                    dp[j][i] = dp[k][j] + 1
                }
                ans = max(ans, dp[j][i])
            }
        }
        return ans > 0 ? ans + 2 : 0
    }

    func fillCups(_ amount: [Int]) -> Int {
        var ans = 0
        var amount = amount.sorted()
        while amount[0] != 0 || amount[1] != 0 || amount[2] != 0 {
            amount.sort()
            ans += 1
            if amount[1] > 0 {
                amount[1] -= 1
            }
            amount[2] -= 1
        }
        return ans
    }

    func canChange(_ start: String, _ target: String) -> Bool {
        let n = start.count
        var cs = [Character](start)
        let ct = [Character](target)
        var i = 0, j = 0
        while i < n && j < n {
            while j < n && ct[j] != "L" {
                j += 1
            }
            while i < n && cs[i] != "L" {
                if i >= j && cs[i] == "R" {
                    return false
                }
                i += 1
            }
            if i >= n || j >= n {
                continue
            }
            if (cs[i] != "L" && ct[j] == "L") || (cs[i] == "L" && ct[j] != "L") || (j > i) {
                return false
            }
            swap(i: i, j: j)
            i += 1
            j += 1
        }
        i = n - 1
        j = n - 1
        while i >= 0 && j >= 0 {
            while j >= 0 && ct[j] != "R" {
                j -= 1
            }
            while i >= 0 && cs[i] != "R" {
                if i <= j && cs[i] == "L" {
                    return false
                }
                i -= 1
            }
            if i < 0 || j < 0 {
                continue
            }
            if (cs[i] != "R" && ct[j] == "R") || (cs[i] == "R" && ct[j] != "R") || (j < i) {
                return false
            }
            swap(i: i, j: j)
            i -= 1
            j -= 1
        }
        func swap(i: Int, j: Int) {
            let temp = cs[i]
            cs[i] = cs[j]
            cs[j] = temp
        }
        return cs == ct
    }

    func findUnsortedSubarray(_ nums: [Int]) -> Int {
        let n = nums.count
        var l = -1, r = -1
        for i in 1..<n where nums[i] <= nums[i - 1] {
            l = i - 1
            break
        }
        for i in (0..<n-1).reversed() where nums[i] >= nums[i + 1] {
            r = i + 1
            break
        }
        return (l == -1 && r == -1) ? 0 : r - l + 1
    }

    func oddCells(_ m: Int, _ n: Int, _ indices: [[Int]]) -> Int {
        var rows = 0, cols = 0
        for indice in indices {
            rows ^= (1 << indice[0])
            cols ^= (1 << indice[1])
        }
        let oddX = rows.nonzeroBitCount, oddY = cols.nonzeroBitCount
        return oddX * (n - oddY) + oddY * (m - oddX)
    }

    func asteroidCollision(_ asteroids: [Int]) -> [Int] {
        var ans = [Int]()
        var stack = [Int]()
        for asteroid in asteroids {
            if asteroid > 0 {
                stack.append(asteroid)
            } else {
                var isAns = true
                while !stack.isEmpty && isAns {
                    if -asteroid < stack.last! {
                        isAns = false
                    } else if -asteroid > stack.last! {
                        stack.removeLast()
                    } else {
                        stack.removeLast()
                        isAns = false
                    }
                }
                if isAns {
                    ans.append(asteroid)
                }
            }
        }
        ans.append(contentsOf: stack)
        return ans
    }

    func intersect(_ quadTree1: FNode?, _ quadTree2: FNode?) -> FNode? {
        func dfs(q1: FNode, q2: FNode) -> FNode {
            if q1.isLeaf && q2.isLeaf {
                return FNode(q1.val || q2.val, true)
            }
            if (q1.isLeaf && q1.val) || (q2.isLeaf && q2.val) {
                return FNode(true, true)
            }
            let topLeft = dfs(q1: q1.topLeft ?? FNode(q1.val, true), q2: q2.topLeft ?? FNode(q2.val, true))
            let topRight = dfs(q1: q1.topRight ?? FNode(q1.val, true), q2: q2.topRight ?? FNode(q2.val, true))
            let bottomLeft = dfs(q1: q1.bottomLeft ?? FNode(q1.val, true), q2: q2.bottomLeft ?? FNode(q2.val, true))
            let bottomRight = dfs(q1: q1.bottomRight ?? FNode(q1.val, true), q2: q2.bottomRight ?? FNode(q2.val, true))
            let a = topLeft.isLeaf && topRight.isLeaf && bottomLeft.isLeaf && bottomRight.isLeaf
            let b = topLeft.val == topRight.val && topLeft.val == bottomLeft.val && topLeft.val == bottomRight.val
            if a && b {
                return FNode(topLeft.val, true)
            }
            let node = FNode(false, false)
            node.topLeft = topLeft
            node.topRight = topRight
            node.bottomLeft = bottomLeft
            node.bottomRight = bottomRight
            return node
        }
        if let q1 = quadTree1, let q2 = quadTree2 {
            return dfs(q1: q1, q2: q2)
        }
        return nil
    }

    func dailyTemperatures(_ temperatures: [Int]) -> [Int] {
        let n = temperatures.count
        var ans = [Int].init(repeating: 0, count: n)
        var stack = [Int]()
        for i in 0..<n {
            while !stack.isEmpty && temperatures[stack.last!] < temperatures[i] {
                ans[stack.last!] = i - stack.last!
                stack.removeLast()
            }
            stack.append(i)
        }
        return ans
    }

    func arrayNesting(_ nums: [Int]) -> Int {
        var memo = [Int: Int]()
        var ans = 0, set = Set<Int>()
        let n = nums.count
        func dfs(i: Int, cnt: Int, set: inout Set<Int>) -> Int {
            if let currCnt = memo[i] {
                return currCnt + cnt
            }
            if set.contains(nums[i]) {
                return cnt
            }
            set.insert(nums[i])
            let nextCnt = dfs(i: nums[i], cnt: cnt + 1, set: &set)
            memo[i] = nextCnt
            set.remove(nums[i])
            return nextCnt
        }
        for i in 0..<n {
            ans = max(dfs(i: i, cnt: 0, set: &set), ans)
        }
        return ans
    }

    func shiftGrid(_ grid: [[Int]], _ k: Int) -> [[Int]] {
        let n = grid.count, m = grid[0].count
        let k = k % (n * m)
        if k == 0 { return grid }
        var start: (Int, Int) = (k / m, k % m)
        var ans = [[Int]].init(repeating: [Int].init(repeating: 0, count: m), count: n)
        for i in 0..<n {
            for j in 0..<m {
                ans[start.0][start.1] = grid[i][j]
                if start.1 == m - 1 {
                    start.0 = (start.0 + 1) % n
                }
                start.1 = (start.1 + 1) % m
            }
        }
        return ans
    }

    func pruneTree(_ root: TreeNode?) -> TreeNode? {
        guard let root = root else { return nil }
        root.left = pruneTree(root.left)
        root.right = pruneTree(root.right)
        if root.left == nil && root.right == nil && root.val == 0 {
            return nil
        }
        return root
    }

    func intersectionSizeTwo(_ intervals: [[Int]]) -> Int {
        let intervals = intervals.sorted { item1, item2 in
            return item1[0] != item2[0] ? item1[0] < item2[0] : item1[1] > item2[1]
        }
        let n = intervals.count
        var curr = intervals[n - 1][0]
        var next = curr + 1
        var ans = 2
        for i in (0..<n-1).reversed() {
            if intervals[i][1] >= next {
                continue
            } else if intervals[i][1] < curr {
                curr = intervals[i][0]
                next = curr + 1
                ans += 2
            } else {
                next = curr
                curr = intervals[i][0]
                ans += 1
            }
        }
        return ans
    }

    func sequenceReconstruction(_ nums: [Int], _ sequences: [[Int]]) -> Bool {
        let n = nums.count
        var cnts = [Int](repeating: 0, count: n)
        for num in nums {
            cnts[num - 1] += 1
        }
        var totals = [Int](repeating: 0, count: n)
        for sequence in sequences {
            var curr = [Int](repeating: 0, count: n)
            for num in sequence {
                curr[num - 1] += 1
            }
            for i in 0..<n {
                totals[i] = max(totals[i], curr[i])
            }
        }
        for i in 0..<n {
            if totals[i] < cnts[i] {
                return false
            }
        }
        return true
    }

    func distanceBetweenBusStops(_ distance: [Int], _ start: Int, _ destination: Int) -> Int {
        let s = min(start, destination)
        let d = max(start, destination)
        let n = distance.count
        var total = 0, clockwise = 0
        for i in 0..<n {
            total += distance[i]
            if (s..<d).contains(i) {
                clockwise += distance[i]
            }
        }
        return min(total - clockwise, clockwise)
    }

    func repeatedCharacter(_ s: String) -> Character {
        var set = Set<Character>()
        for char in s {
            if set.contains(char) {
                return char
            }
            set.insert(char)
        }
        return "c"
    }

    func equalPairs(_ grid: [[Int]]) -> Int {
        let n = grid.count
        var cols = [String]()
        var rows = [String]()
        for i in 0..<n {
            var col = ""
            var row = ""
            for j in 0..<n {
                col += "\(grid[i][j])-"
                row += "\(grid[j][i])-"
            }
            cols.append(col)
            rows.append(row)
        }
        var ans = 0
        for i in 0..<n {
            for j in 0..<n {
                if cols[i] == rows[j] {
                    ans += 1
                }
            }
        }
        return ans
    }

    func fractionAddition(_ expression: String) -> String {
        var arr = [(Int, Int)]()
        var operations = [Character]()
        let expression = [Character](expression)
        let n = expression.count
        var i = 0
        while i < n {
            var item = (0, 0)
            var num = ""
            while i < n && expression[i] != "/" {
                num.append(expression[i])
                i += 1
            }
            item.0 = Int(num)!
            num = ""
            i += 1
            while i < n && expression[i] != "+" && expression[i] != "-" {
                num.append(expression[i])
                i += 1
            }
            item.1 = Int(num)!
            arr.append(item)
            if i < n {
                operations.append(expression[i])
            }
            i += 1
        }
        for operation in operations {
            let item1 = arr.removeFirst()
            let item2 = arr.removeFirst()
            let x1 = item1.0, x2 = item2.0
            let y1 = item1.1, y2 = item2.1
            var x3 = 0, y3 = y1 * y2
            if operation == "+" {
                x3 = x1 * y2 + x2 * y1
            } else {
                x3 = x1 * y2 - x2 * y1
            }
            let d = gcd(a: abs(x3), b: abs(y3))
            arr.insert((x3 / d, y3 / d), at: 0)
        }
        func gcd(a: Int, b: Int) -> Int {
            return b == 0 ? a : gcd(a: b, b: a % b)
        }
        func abs(_ x: Int) -> Int {
            return x > 0 ? x : -x
        }
        return "\(arr.first!.0)/\(arr.first!.1)"
    }

    func arrayRankTransform(_ arr: [Int]) -> [Int] {
        let n = arr.count
        let sortedArr = arr.sorted()
        var map = [Int : Int]()
        var curr = 0
        for num in sortedArr {
            if !map.keys.contains(num) {
                map[num] = curr
                curr += 1
            }
        }
        var ans = [Int](repeating: 0, count: n)
        for (i, num) in arr.enumerated() {
            ans[i] = map[num]! + 1
        }
        return ans
    }

    func validSquare(_ p1: [Int], _ p2: [Int], _ p3: [Int], _ p4: [Int]) -> Bool {
        var len = -1
        func calu(a: [Int], b: [Int], c: [Int]) -> Bool {
            let l1 = (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1])
            let l2 = (a[0] - c[0]) * (a[0] - c[0]) + (a[1] - c[1]) * (a[1] - c[1])
            let l3 = (b[0] - c[0]) * (b[0] - c[0]) + (b[1] - c[1]) * (b[1] - c[1])
            let ok = (l1 == l2 && l1 + l2 == l3) || (l1 == l3 && l1 + l3 == l2) || (l3 == l2 && l3 + l2 == l1)
            if !ok { return false }
            if len == -1 { len = min(l1, l2) }
            else if len == 0 || len != min(l1, l2) { return false }
            return true
        }
        return calu(a: p1, b: p2, c: p3) && calu(a: p1, b: p2, c: p4) && calu(a: p1, b: p3, c: p4) && calu(a: p2, b: p3, c: p4)
    }

    func maxLevelSum(_ root: TreeNode?) -> Int {
        guard let root = root else { return 0 }
        var queue: [TreeNode] = [root]
        var ans = -1, maxSum = Int.min, floor = 1
        while !queue.isEmpty {
            let n = queue.count
            var floorSum = 0
            for _ in 0..<n {
                let curr = queue.removeFirst()
                floorSum += curr.val
                if let left = curr.left { queue.append(left) }
                if let right = curr.right { queue.append(right) }
            }
            if maxSum < floorSum {
                ans = floor
                maxSum = floorSum
            }
            floor += 1
        }
        return ans
    }

    func generateTheString(_ n: Int) -> String {
        var ans = ""
        ans.append(contentsOf: [Character].init(repeating: "a", count: n | 1 - 1))
        if n & 1 == 0 { ans.append("b") }
        return ans
    }

    func orderlyQueue(_ s: String, _ k: Int) -> String {
        var ans = s
        let n = s.count
        if k == 1 {
            var chars: [Character] = [Character].init(s)
            for _ in 0..<n {
                let char = chars.removeFirst()
                chars.append(char)
                let curr = String(chars)
                if ans > curr {
                    ans = curr
                }
            }
        } else if k > 1 { return String(s.sorted()) }
        return ans
    }

    func minSubsequence(_ nums: [Int]) -> [Int] {
        let nums = nums.sorted()
        var sum = nums.reduce(0, { $0 + $1 })
        let n = nums.count
        var currSum = 0, ans = [Int](), i = n - 1
        while currSum < sum {
            ans.append(nums[i])
            currSum += nums[i]
            sum -= nums[i]
            i -= 1
        }
        return ans
    }

    func addOneRow(_ root: TreeNode?, _ val: Int, _ depth: Int) -> TreeNode? {
        guard let root = root else { return nil }
        if depth == 1 {
            return TreeNode(val, root, nil)
        }
        var currDepth = 1
        var queue: [TreeNode] = [root]
        while !queue.isEmpty {
            let n = queue.count
            for _ in 0..<n {
                let curr = queue.removeFirst()
                let left = curr.left
                let right = curr.right
                if currDepth == depth - 1 {
                    curr.left = TreeNode(1, left, nil)
                    curr.right = TreeNode(1, nil, right)
                    continue
                }
                if let left = curr.left {
                    queue.append(left)
                }
                if let right = curr.right {
                    queue.append(right)
                }
            }
            currDepth += 1
        }
        return root
    }

    func stringMatching(_ words: [String]) -> [String] {
        let words = words.sorted(by: { $0.count < $1.count })
        let n = words.count
        var ans = [String]()
        for i in 0..<n-1 {
            for j in i+1..<n {
                if words[j].contains(words[i]) {
                    ans.append(words[i])
                    break
                }
            }
        }
        return ans
    }

    func exclusiveTime(_ n: Int, _ logs: [String]) -> [Int] {
        var ans = [Int](repeating: 0, count: n)
        var stack = [Int]()
        var curr = -1
        for log in logs {
            let arr = log.split(separator: ":")
            let idx = Int(arr[0])!, ts = Int(arr[2])!
            if arr[1] == "start" {
                if !stack.isEmpty {
                    ans[stack.last!] += (ts - curr)
                }
                stack.append(idx)
                curr = ts
            } else {
                let last = stack.popLast()!
                ans[last] += (ts - curr + 1)
                curr = ts + 1
            }
        }
        return ans
    }

    func minStartValue(_ nums: [Int]) -> Int {
        var minSum = 0
        var sum = 0
        for num in nums {
            sum += num
            if sum < 0 {
                minSum = min(minSum, sum)
            }
        }
        return -minSum + 1
    }


}

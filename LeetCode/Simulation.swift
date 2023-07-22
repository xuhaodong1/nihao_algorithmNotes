//
//  Simulation.swift
//  LeetCode
//
//  Created by haodong xu on 2022/9/4.
//

import Foundation

/// 题目链接：[1797. 设计一个验证系统](https://leetcode.cn/problems/design-authentication-manager/description/)
class AuthenticationManager {
    let timeToLive: Int
    var map = [String: (Int, Int)]()
    init(_ timeToLive: Int) {
        self.timeToLive = timeToLive
    }
    
    func generate(_ tokenId: String, _ currentTime: Int) {
        map[tokenId] = (currentTime, currentTime + timeToLive)
    }
    
    func renew(_ tokenId: String, _ currentTime: Int) {
        if map.keys.contains(tokenId) && map[tokenId]!.1 > currentTime {
            map[tokenId] = (currentTime, currentTime + timeToLive)
        }
    }
    
    func countUnexpiredTokens(_ currentTime: Int) -> Int {
        map = map.filter { $0.value.1 > currentTime }
        return map.count
    }
}

/// 模拟相关练习题
class Simulation: BaseCode {
    
    /// 题目链接：[1582. 二进制矩阵中的特殊位置](https://leetcode.cn/problems/special-positions-in-a-binary-matrix/)
    func numSpecial(_ mat: [[Int]]) -> Int {
        let n = mat.count, m = mat[0].count
        var cols = [Int](repeating: 0, count: n)
        var rows = [Int](repeating: 0, count: m)
        var ans = 0
        for i in 0..<n {
            for j in 0..<m where mat[i][j] == 1 {
                cols[i] += 1; rows[j] += 1
            }
        }
        for i in 0..<n {
            for j in 0..<m where mat[i][j] == 1 && cols[i] == 1 && rows[j] == 1 {
                ans += 1
                break
            }
        }
        return ans
    }
    
    /// 题目链接：[1598. 文件夹操作日志搜集器](https://leetcode.cn/problems/crawler-log-folder/)
    func minOperations(_ logs: [String]) -> Int {
        return logs.reduce(0) { partialResult, log in
            if log == "../" {
                return max(partialResult - 1, 0)
            } else if log == "./" {
                return partialResult
            } else {
                return partialResult + 1
            }
        }
    }
    
    /// 题目链接：[1619. 删除某些元素后的数组均值](https://leetcode.cn/problems/mean-of-array-after-removing-some-elements/)
    func trimMean(_ arr: [Int]) -> Double {
        let n = arr.count, removeCnt = Int(Double(n) * 0.05)
        return Double(arr.sorted()[removeCnt..<n-removeCnt].reduce(0, +)) / Double(n - removeCnt * 2)
    }
    
    /// 题目链接：[1624. 两个相同字符之间的最长子字符串](https://leetcode.cn/problems/largest-substring-between-two-equal-characters/)
    func maxLengthBetweenEqualCharacters(_ s: String) -> Int {
        var map = [Character: Int](), ans = -1
        for (index, char) in s.enumerated() {
            if let l = map[char] {
                ans = max(index - l - 1, ans)
            } else {
                map[char] = index
            }
        }
        return ans
    }
    
    /// 题目链接：[1636. 按照频率将数组升序排序](https://leetcode.cn/problems/sort-array-by-increasing-frequency/)
    func frequencySort(_ nums: [Int]) -> [Int] {
        return [Int: Int].init(nums.map{($0, 1)}, uniquingKeysWith: +).sorted { kv1, kv2 in
            return kv1.value != kv2.value ? kv1.value < kv2.value : kv1.key > kv2.key
        }.map { [Int](repeating: $0.key, count: $0.value) }.flatMap{ $0 }
    }
    
    /// 题目链接：[1640. 能否连接形成数组](https://leetcode.cn/problems/check-array-formation-through-concatenation/)
    func canFormArray(_ arr: [Int], _ pieces: [[Int]]) -> Bool {
        let n = arr.count
        var i = 0, map = [Int: [Int]]()
        pieces.forEach { item in map[item[0]] = item }
        while i < n {
            guard let item = map[arr[i]] else { return false }
            for num in item where i < n {
                if num != arr[i] { return false }
                i += 1
            }
        }
        return true
    }
    
    /// 题目链接：[788. 旋转数字](https://leetcode.cn/problems/rotated-digits/)
    func rotatedDigits(_ n: Int) -> Int {
        var cnt = 0
        for i in 1...n where isGoodNum(i) { cnt += 1 }
        func isGoodNum(_ x: Int) -> Bool {
            var x = x, hasNeed = false
            while x != 0 {
                let digit = x % 10
                if digit == 3 || digit == 4 || digit == 7 { return false } /// 不包含3 4 7
                if digit == 2 || digit == 5 || digit == 6 || digit == 9 { hasNeed = true } /// 必须包含 2 / 5 / 6 / 9
                x /= 10
            }
            return hasNeed
        }
        return cnt
    }
    
    /// 题目链接：[面试题 01.02. 判定是否互为字符重排](https://leetcode.cn/problems/check-permutation-lcci/)
    func CheckPermutation(_ s1: String, _ s2: String) -> Bool {
        guard s1.count == s2.count else { return false }
        var map = [Character: Int]()
        s1.forEach{ map[$0, default: 0] += 1 }
        s2.forEach{ map[$0, default: 0] -= 1 }
        return map.filter{ $0.value != 0 }.count == 0
    }
    
    /// 题目链接：[面试题 01.09. 字符串轮转](https://leetcode.cn/problems/string-rotation-lcci/)
    func isFlipedString(_ s1: String, _ s2: String) -> Bool {
        guard s1.count == s2.count else { return false }
        guard s1.count != 0 else { return true }
        return (s1 + s1).contains(s2)
    }
    
    /// 题目链接：[面试题 01.08. 零矩阵](https://leetcode.cn/problems/zero-matrix-lcci/)
    func setZeroes(_ matrix: inout [[Int]]) {
        let n = matrix.count, m = matrix[0].count
        var flag_col = false
        for i in 0..<n {
            if matrix[i][0] == 0 { flag_col = true }
            for j in 1..<m where matrix[i][j] == 0 {
                matrix[i][0] = 0
                matrix[0][j] = 0
            }
        }
        for i in (0..<n).reversed() {
            for j in 1..<m where matrix[i][0] == 0 || matrix[0][j] == 0 {
                matrix[i][j] = 0
            }
            if flag_col { matrix[i][0] = 0 }
        }
    }
    
    /// 题目链接：[1694. 重新格式化电话号码](https://leetcode.cn/problems/reformat-phone-number/)
    func reformatNumber(_ number: String) -> String {
        let chars = [Character](number.compactMap{ $0.isNumber ? $0 : nil })
        var cnt = chars.count, curr = 0
        let last = cnt % 3 == 1 ? 4 : cnt % 3
        var ans = [Character]()
        while curr < cnt {
            if last == 4 && curr + last == cnt {
                ans.append(contentsOf: [chars[curr], chars[curr + 1], "-", chars[curr + 2], chars[curr + 3]])
                curr += last
            } else if last == 2 && curr + last == cnt {
                ans.append(contentsOf: [chars[curr], chars[curr + 1]])
                curr += last
            } else {
                ans.append(contentsOf: [chars[curr], chars[curr + 1], chars[curr + 2]])
                curr += 3
            }
            ans.append("-")
        }
        ans.removeLast()
        return String(ans)
    }
    
    /// 题目链接：[1784. 检查二进制字符串字段](https://leetcode.cn/problems/check-if-binary-string-has-at-most-one-segment-of-ones/)
    func checkOnesSegment(_ s: String) -> Bool {
        let n = s.count, chars = [Character](s)
        for i in 1..<n where chars[i] == "1" && chars[i - 1] == "0" {
            return false
        }
        return true
    }
    
    /// 题目链接：[811. 子域名访问计数](https://leetcode.cn/problems/subdomain-visit-count/)
    func subdomainVisits(_ cpdomains: [String]) -> [String] {
        var map = [String: Int]()
        cpdomains.forEach { cpdomain in
            let arr = cpdomain.split(separator: " ")
            guard let cnt = Int(arr[0]) else { return }
            var curr = ""
            arr[1].split(separator: ".").reversed().forEach { item in
                curr = item + curr
                map[curr, default: 0] += cnt
                curr = "." + curr
            }
        }
        return map.map { return "\($0.value) \($0.key)" }
    }
    
    /// 题目链接：[921. 使括号有效的最少添加](https://leetcode.cn/problems/minimum-add-to-make-parentheses-valid/)
    func minAddToMakeValid(_ s: String) -> Int {
        var score = 0, offset = 0
        s.forEach { char in
            score += (char == "(" ? 1 : -1)
            if score < 0 {
                score = 0
                offset += 1
            }
        }
        return score + offset
    }
    
    /// 题目链接：[927. 三等分](https://leetcode.cn/problems/three-equal-parts/)
    func threeEqualParts(_ arr: [Int]) -> [Int] {
        let n = arr.count, cnt = arr.reduce(0, +)
        guard cnt % 3 == 0 else { return [-1, -1] }
        guard cnt != 0 else { return [0, n - 1] }
        var i = 0, j = 0, k = 0, currCnt = 0
        for (index, num) in arr.enumerated() {
            currCnt += num
            if currCnt == 1 && num == 1 { i = index }
            else if currCnt == cnt / 3 + 1 && num == 1 { j = index }
            else if currCnt == cnt / 3 * 2 + 1 && num == 1 { k = index }
        }
        while k < n {
            if arr[i] != arr[j] || arr[j] != arr[k] { return [-1, -1] }
            i += 1; j += 1; k += 1
        }
        return [i - 1, j]
    }
    
    /// 题目链接：[1800. 最大升序子数组和](https://leetcode.cn/problems/maximum-ascending-subarray-sum/)
    func maxAscendingSum(_ nums: [Int]) -> Int {
        let n = nums.count
        var ans = nums[0], curr = nums[0]
        for i in 1..<n {
            if nums[i] > nums[i - 1] { curr += nums[i] }
            else { curr = nums[i] }
            ans = max(ans, curr)
        }
        return ans
    }
    
    /// 题目链接：[856. 括号的分数](https://leetcode.cn/problems/score-of-parentheses/)
    func scoreOfParentheses(_ s: String) -> Int {
        var stack = [0]
        for char in s {
            if char == "(" {
                stack.append(0)
            } else {
                let curr = stack.removeLast()
                stack.append(stack.removeLast() + max(curr * 2, 1))
            }
        }
        return stack.removeLast()
    }
    
    /// 题目链接：[1790. 仅执行一次字符串交换能否使两个字符串相等](https://leetcode.cn/problems/check-if-one-string-swap-can-make-strings-equal/)
    func areAlmostEqual(_ s1: String, _ s2: String) -> Bool {
        let n = s1.count
        var chars1 = [Character](s1), chars2 = [Character](s2)
        var diff = [Int]()
        for i in 0..<n where chars1[i] != chars2[i] {
            diff.append(i)
            if diff.count > 2 { return false }
        }
        if diff.count == 0 { return true }
        if diff.count != 2 { return false }
        chars1.swapAt(diff[0], diff[1])
        return chars1 == chars2
    }
    
    /// 题目链接：[817. 链表组件](https://leetcode.cn/problems/linked-list-components/)
    func numComponents(_ head: ListNode?, _ nums: [Int]) -> Int {
        let set = Set<Int>(nums)
        var ans = 0, head = head
        while head != nil {
            if set.contains(head!.val) {
                ans += 1
                while head != nil && set.contains(head!.val) {
                    head = head?.next
                }
            } else {
                head = head?.next
            }
        }
        return ans
    }
    
    /// 题目链接：[769. 最多能完成排序的块](https://leetcode.cn/problems/max-chunks-to-make-sorted/)
    func maxChunksToSorted(_ arr: [Int]) -> Int {
        var ma = 0, ans = 0
        for (i, num) in arr.enumerated() {
            ma = max(num, ma)
            if ma == i { ans += 1 }
        }
        return ans
    }
    
    /// 题目链接：[1441. 用栈操作构建数组](https://leetcode.cn/problems/build-an-array-with-stack-operations/)
    func buildArray(_ target: [Int], _ n: Int) -> [String] {
        var ans = [String]()
        var curr = 1
        for num in target {
            while curr < num {
                ans.append("Push")
                ans.append("Pop")
                curr += 1
            }
            if curr == num {
                ans.append("Push")
                curr += 1
            }
        }
        return ans
    }
    
    
    /// 题目链接：[1700. 无法吃午餐的学生数量](https://leetcode.cn/problems/number-of-students-unable-to-eat-lunch/)
    func countStudents(_ students: [Int], _ sandwiches: [Int]) -> Int {
        var circleCnt = 0, squareCnt = 0
        var currIndex = 0
        students.forEach { curr in
            if curr == 0 { circleCnt += 1 }
            else { squareCnt += 1 }
        }
        for sandwich in sandwiches {
            if (sandwich == 0 && circleCnt == 0) || (sandwich == 1 && squareCnt == 0) { break }
            else if sandwich == 0 { circleCnt -= 1 }
            else { squareCnt -= 1 }
            currIndex += 1
        }
        return sandwiches.count - currIndex
    }
    
    /// 题目链接：[779. 第K个语法符号](https://leetcode.cn/problems/k-th-symbol-in-grammar/)
    func kthGrammar(_ n: Int, _ k: Int) -> Int {
        if k == 1 { return 0 }
        if k > 1 << (n - 2) { return 1 ^ kthGrammar(n - 1, k - 1 << (n - 2))}
        else { return kthGrammar(n - 1, k) }
    }
    
    /// 题目链接：[1768. 交替合并字符串](https://leetcode.cn/problems/merge-strings-alternately/)
    func mergeAlternately(_ word1: String, _ word2: String) -> String {
        let n = word1.count, m = word2.count
        let word1 = [Character](word1), word2 = [Character](word2)
        var ans = [Character]()
        for i in 0..<min(n, m) {
            ans.append(contentsOf: [word1[i], word2[i]])
        }
        if n > m { ans.append(contentsOf: word1[m..<n]) }
        else if m > n { ans.append(contentsOf: word2[n..<m]) }
        return String(ans)
    }
    
    /// 题目链接：[1773. 统计匹配检索规则的物品数量](https://leetcode.cn/problems/count-items-matching-a-rule/)
    func countMatches(_ items: [[String]], _ ruleKey: String, _ ruleValue: String) -> Int {
        var cnt = 0
        for item in items {
            if ruleKey == "type" && item[0] == ruleValue { cnt += 1 }
            else if ruleKey == "color" && item[1] == ruleValue { cnt += 1 }
            else if ruleKey == "name" && item[2] == ruleValue { cnt += 1 }
        }
        return cnt
    }
    
    /// 题目链接：[915. 分割数组](https://leetcode.cn/problems/partition-array-into-disjoint-intervals/)
    func partitionDisjoint(_ nums: [Int]) -> Int {
        let n = nums.count
        var minArr = [Int](repeating: 0, count: n)
        minArr[n - 1] = nums[n - 1]
        for i in (0...n-2).reversed() {
            minArr[i] = min(minArr[i + 1], nums[i])
        }
        var maxV = 0
        for i in 0...n-2 {
            maxV = max(maxV, nums[i])
            if maxV <= minArr[i + 1] { return i + 1 }
        }
        return -1
    }
    
    /// 题目链接：[1822. 数组元素积的符号](https://leetcode.cn/problems/sign-of-the-product-of-an-array/)
    func arraySign(_ nums: [Int]) -> Int {
        var negativeCnt = 0
        for num in nums {
            if num == 0 { return 0 }
            if num < 0 { negativeCnt += 1 }
        }
        return negativeCnt & 1 == 0 ? 1 : -1
    }
    
    /// 题目链接：[1662. 检查两个字符串数组是否相等](https://leetcode.cn/problems/check-if-two-string-arrays-are-equivalent/)
    func arrayStringsAreEqual(_ word1: [String], _ word2: [String]) -> Bool {
        return word1.reduce("", +) == word2.reduce("", +)
    }
    
    /// 题目链接：[1620. 网络信号最好的坐标](https://leetcode.cn/problems/coordinate-with-maximum-network-quality/)
    func bestCoordinate(_ towers: [[Int]], _ radius: Int) -> [Int] {
        var minX = 50, minY = 50, maxX = 0, maxY = 0
        for tower in towers {
            minX = min(tower[0], minX)
            minY = min(tower[1], minY)
            maxX = max(tower[0], maxX)
            maxY = max(tower[1], maxY)
        }
        var point = [0, 0], maxS = 0.0
        for x in minX...maxX {
            for y in minY...maxY {
                var curr = 0.0
                for tower in towers {
                    let d = getSignalStrength(tower: (tower[0], tower[1]), point: (x, y))
                    if d <= radius * radius {
                        let s = floor(Double(tower[2]) / (1 + sqrt(Double(d))))
                        curr += s
                    }
                }
                if curr > maxS {
                    maxS = curr
                    point = [x, y]
                }
            }
        }
        func getSignalStrength(tower: (Int, Int), point: (Int, Int)) -> Int {
            return (tower.0 - point.0) * (tower.0 - point.0) + (tower.1 - point.1) * (tower.1 - point.1)
        }
        return point
    }
    
    /// 题目链接：[1678. 设计 Goal 解析器](https://leetcode.cn/problems/goal-parser-interpretation/description/)
    func interpret(_ command: String) -> String {
        let n = command.count, chars = [Character](command)
        var i = 0, ans = [String]()
        while i < n {
            if chars[i] == "G" {
                ans.append("G")
                i += 1
            } else if i + 1 < n && chars[i + 1] == ")" {
                ans.append("o")
                i += 2
            } else {
                ans.append("al")
                i += 4
            }
        }
        return ans.reduce("", +)
    }
    
    /// 题目链接：[816. 模糊坐标](https://leetcode.cn/problems/ambiguous-coordinates/description/)
    func ambiguousCoordinates(_ s: String) -> [String] {
        let chars = [Character](s.filter{ $0 != "(" && $0 != ")" }), n = chars.count
        var ans = [String]()
        for i in 0..<n-1 {
            let pre = getVaildStr(0, end: i), suf = getVaildStr(i + 1 , end: n - 1)
            for item1 in pre {
                for item2 in suf {
                    ans.append("(\(item1), \(item2))")
                }
            }
        }
        func getVaildStr(_ start: Int, end: Int) -> [String] {
            if start == end { return ["\(chars[start])"] }
            var ans = [String]()
            let str = (start...end).reduce("") { $0 + "\(chars[$1])" }
            if str.first! != "0" { ans.append(str) }
            if str.last! == "0" { return ans }
            for i in 1..<str.count {
                let curr = "\(str.prefix(i)).\(str.suffix(str.count - i))"
                if i > 1 && curr.first! == "0" { break }
                ans.append(curr)
            }
            return ans
        }
        return ans
    }
    
    /// 题目链接：[1684. 统计一致字符串的数目](https://leetcode.cn/problems/count-the-number-of-consistent-strings/)
    func countConsistentStrings(_ allowed: String, _ words: [String]) -> Int {
        let set = Set<Character>(allowed), n = words.count
        return n - words.filter { $0.filter { !set.contains($0) }.count > 0 }.count
    }
    
    /// 题目链接：[1704. 判断字符串的两半是否相似](https://leetcode.cn/problems/determine-if-string-halves-are-alike/description/)
    func halvesAreAlike(_ s: String) -> Bool {
        let halfLength = s.count / 2
        let vowels: [Character] = ["a", "e", "i", "o", "u", "A", "E", "I", "O", "U"]
        func handleChar(_ ans: Int, _ char: Character) -> Int { return vowels.contains(char) ? ans + 1 : ans }
        return s.prefix(halfLength).reduce(0, handleChar(_:_:)) == s.suffix(halfLength).reduce(0, handleChar(_:_:))
    }
    
    /// 题目链接：[791. 自定义字符串排序](https://leetcode.cn/problems/custom-sort-string/description/)
    func customSortString(_ order: String, _ s: String) -> String {
        var map = [Character: Int](), index = order.startIndex
        (1...order.count).reversed().forEach { weight in
            map[order[index]] = weight
            index = order.index(after: index)
        }
        return String(s.sorted(by: { return map[$0, default: 0] > map[$1, default: 0] }))
    }
    
    /// 题目链接：[1732. 找到最高海拔](https://leetcode.cn/problems/find-the-highest-altitude/)
    func largestAltitude(_ gain: [Int]) -> Int {
        var ans = 0, last = 0
        gain.forEach { diff in
            last = last + diff
            ans = max(ans, last)
        }
        return ans
    }
    
    /// 题目链接：[1742. 盒子中小球的最大数量](https://leetcode.cn/problems/maximum-number-of-balls-in-a-box/)
    func countBalls(_ lowLimit: Int, _ highLimit: Int) -> Int {
        var boxes = [Int](repeating: 0, count: 50), ans = 0
        for i in lowLimit...highLimit {
            var i = i, sum = 0
            while i > 0 {
                sum += i % 10
                i /= 10
            }
            boxes[sum] += 1
            ans = max(ans, boxes[sum])
        }
        return ans
    }
    
    /// 题目链接：[1752. 检查数组是否经排序和轮转得到](https://leetcode.cn/problems/check-if-array-is-sorted-and-rotated/description/)
    func check(_ nums: [Int]) -> Bool {
        let n = nums.count
        let cnt = (0..<n).reduce(0) { cnt, i in return cnt + nums[i] < nums[(i + 1) % n] ? 1 : 0 }
        return cnt <= 1
    }
    
    /// 题目链接：[1758. 生成交替二进制字符串的最少操作数](https://leetcode.cn/problems/minimum-changes-to-make-alternating-binary-string/)
    func minOperations(_ s: String) -> Int {
        var cnt = 0
        for (i, c) in s.enumerated() {
            if i & 1 == 0 { cnt += (c == "0" ? 0 : 1) }
            else { cnt += (c == "0" ? 1 : 0) }
        }
        return min(cnt, s.count - cnt)
    }
    
    /// 题目链接：[1779. 找到最近的有相同 X 或 Y 坐标的点](https://leetcode.cn/problems/find-nearest-point-that-has-the-same-x-or-y-coordinate/description/)
    func nearestValidPoint(_ x: Int, _ y: Int, _ points: [[Int]]) -> Int {
        var ans = -1, len = Int.max
        for (i, point) in points.enumerated() where (point[0] == x || point[1] == y) {
            let currLen = abs(point[0] - x) + abs(point[1] - y)
            if currLen < len {
                ans = i
                len = currLen
            }
        }
        return ans
    }
    
    /// 题目链接：[1769. 移动所有球到每个盒子所需的最小操作数](https://leetcode.cn/problems/minimum-number-of-operations-to-move-all-balls-to-each-box/)
    func minOperations(boxes: String) -> [Int] {
        let n = boxes.count, boxes = [Character](boxes)
        var ans = [Int](repeating: 0, count: n)
        var len = 0, rightCnt = 0
        for i in 0..<n where boxes[i] == "1" {
            len += i
            rightCnt += 1
        }
        var leftCnt = 0
        for i in 0..<n {
            ans[i] = len
            if boxes[i] == "1" {
                rightCnt -= 1
                leftCnt += 1
            }
            len = len - rightCnt + leftCnt
        }
        return ans
    }
    
    /// 题目链接：[1796. 字符串中第二大的数字](https://leetcode.cn/problems/second-largest-digit-in-a-string/description/)
    func secondHighest(_ s: String) -> Int {
        var first = -1, second = -1
        for num in s.compactMap({ $0.wholeNumberValue }) {
            if num > first {
                second = first
                first = num
            } else if num < first && num > second {
                second = num
            }
        }
        return second
    }
    
    /// 题目链接：[1805. 字符串中不同整数的数目](https://leetcode.cn/problems/number-of-different-integers-in-a-string/)
    func numDifferentIntegers(_ word: String) -> Int {
        let n = word.count
        var set = Set<String>(), curr = ""
        for (i, c) in word.enumerated() {
            if c.isNumber {
                curr.append(c)
            }
            if i == n - 1 || !c.isNumber, curr != "" {
                set.insert(curr)
                curr = ""
            }
        }
        return Set<String>(set.map { str in
            var str = str
            while str.count > 1 && str.first == "0" {
                str.removeFirst()
            }
            return str
        }).count
    }
    
    /// 题目链接：[1775. 通过最少操作次数使数组的和相等](https://leetcode.cn/problems/equal-sum-arrays-with-minimum-number-of-operations/description/)
    func minOperations(_ nums1: [Int], _ nums2: [Int]) -> Int {
        if nums1.count > nums2.count * 6 || nums1.count * 6 < nums2.count {
            return -1
        }
        var nums1 = nums1, nums2 = nums2
        var d = nums2.reduce(0, +) - nums1.reduce(0, +)
        if d < 0 {
            d = -d
            let temp = nums1
            nums1 = nums2
            nums2 = temp
        }
        var cnts = [Int](repeating: 0, count: 6), ans = 0
        for num in nums1 { cnts[6 - num] += 1 }
        for num in nums2 { cnts[num - 1] += 1 }
        for i in (1...5).reversed() {
            if i * cnts[i] >= d { return ans + (d + i - 1) / i }  // 向上取整
            ans += cnts[i]
            d -= i * cnts[i]
        }
        return ans
    }
    
    /// 题目链接：[1827. 最少操作使数组递增](https://leetcode.cn/problems/minimum-operations-to-make-the-array-increasing/)
    func minOperations(_ nums: [Int]) -> Int {
        var pre = nums[0] - 1
        return nums.reduce(0) { cnt, num in
            pre = max(pre + 1, num)
            return cnt + pre - num
        }
    }
    
    /// 题目链接：[1781. 所有子字符串美丽值之和](https://leetcode.cn/problems/sum-of-beauty-of-all-substrings/description/)
    func beautySum(_ s: String) -> Int {
        let n = s.count, chars = [Character](s)
        let aV = Character("a").asciiValue!
        var ans = 0
        for i in 0..<n {
            var cnts = [Int](repeating: 0, count: 26)
            for j in i..<n {
                var minCnt = Int.max, maxCnt = Int.min
                cnts[Int(chars[j].asciiValue! - aV)] += 1
                for cnt in cnts where cnt > 0 {
                    minCnt = min(minCnt, cnt)
                    maxCnt = max(maxCnt, cnt)
                }
                ans += (maxCnt - minCnt)
            }
        }
        return ans
    }
    
    /// 题目链接：[1832. 判断句子是否为全字母句](https://leetcode.cn/problems/check-if-the-sentence-is-pangram/)
    func checkIfPangram(_ sentence: String) -> Bool {
        var letterCnts = [Bool](repeating: false, count: 26)
        let av = Int(Character("a").asciiValue!)
        for c in sentence { letterCnts[Int(c.asciiValue!) - av] = true }
        return letterCnts.allSatisfy { $0 }
    }
    
    /// 题目链接：[1945. 字符串转化后的各位数字之和](https://leetcode.cn/problems/sum-of-digits-of-string-after-convert/)
    func getLucky(_ s: String, _ k: Int) -> Int {
        let av = Character("a").asciiValue!
        var ans = s.map { c in return "\(c.asciiValue! - av)" }.reduce("", +).map { "\($0)" }
        for _ in 0..<k {
            ans = "\(ans.reduce(0) { sum, c in return sum + Int(c)! })".map({ "\($0)" })
        }
        return Int(ans.reduce("", +)) ?? -1
    }
    
    /// 题目链接：[1785. 构成特定和需要添加的最少元素](https://leetcode.cn/problems/minimum-elements-to-add-to-form-a-given-sum/description/)
    func minElements(_ nums: [Int], _ limit: Int, _ goal: Int) -> Int {
        return (abs(goal - nums.reduce(0, +)) + limit - 1) / limit
    }
    
    /// 题目链接：[1764. 通过连接另一个数组的子数组得到一个数组](https://leetcode.cn/problems/form-array-by-concatenating-subarrays-of-another-array/)
    func canChoose(_ groups: [[Int]], _ nums: [Int]) -> Bool {
        let n = nums.count
        var i = 0, k = 0
        while k < n && i < groups.count {
            if check(i: i, k: k) {
                k += groups[i].count
                i += 1
            } else {
                k += 1
            }
        }
        func check(i: Int, k: Int) -> Bool {
            let m = groups[i].count
            if k + m > n { return false }
            for j in 0..<m { if groups[i][j] != nums[k + j] { return false } }
            return true
        }
        return i == groups.count
    }
    
    /// 题目链接：[2011. 执行操作后的变量值](https://leetcode.cn/problems/final-value-of-variable-after-performing-operations/)
    func finalValueAfterOperations(_ operations: [String]) -> Int {
        return operations.reduce(0) { return $0 + ($1.hasPrefix("+") || $1.hasSuffix("+") ? 1 : -1) }
    }
    
    ///  题目链接：[1754. 构造字典序最大的合并字符串](https://leetcode.cn/problems/largest-merge-of-two-strings/)
    func largestMerge(_ word1: String, _ word2: String) -> String {
        let word1 = word1, word2 = word2
        var i = word1.startIndex, iEnd = word1.endIndex
        var j = word2.startIndex, jEnd = word2.endIndex
        var ans = ""
        while i < iEnd || j < jEnd {
            if word1[i..<iEnd] > word2[j..<jEnd] {
                ans.append(word1[i])
                i = word1.index(after: i)
            } else {
                ans.append(word2[j])
                j = word2.index(after: j)
            }
        }
        return ans
    }
    
    /// 题目链接：[2037. 使每位学生都有座位的最少移动次数](https://leetcode.cn/problems/minimum-number-of-moves-to-seat-everyone/)
    func minMovesToSeat(_ seats: [Int], _ students: [Int]) -> Int {
        let n = seats.count
        let seats = seats.sorted(), students = students.sorted()
        return (0..<n).reduce(0) { $0 + abs(seats[$1] - students[$1]) }
    }
    
    /// 题目链接：[2042. 检查句子中的数字是否递增](https://leetcode.cn/problems/check-if-numbers-are-ascending-in-a-sentence/)
    func areNumbersAscending(_ s: String) -> Bool {
        let chars = [Character](s), n = chars.count
        var lastV = 0, i = 0
        while i < n {
            if !chars[i].isNumber {
                i += 1
            } else {
                var curr = 0
                while i < n && chars[i].isNumber {
                    curr = curr * 10 + chars[i].wholeNumberValue!
                    i += 1
                }
                if curr <= lastV { return false }
                lastV = curr
            }
        }
        return true
    }
    
    /// 题目链接：[2351. 第一个出现两次的字母](https://leetcode.cn/problems/first-letter-to-appear-twice/)
    func repeatedCharacter(_ s: String) -> Character {
        let a = Character("a")
        var mask = 0
        for c in s {
            let i = c.asciiValue! - a.asciiValue!
            if mask >> i & 1 == 1 { return c }
            mask |= (1 << i)
        }
        return "."
    }
    
    /// 题目链接：[1669. 合并两个链表](https://leetcode.cn/problems/merge-in-between-linked-lists/)
    func mergeInBetween(_ list1: ListNode?, _ a: Int, _ b: Int, _ list2: ListNode?) -> ListNode? {
        let dump = ListNode(-1, list1)
        var head: ListNode? = dump
        for _ in 0..<a {
            head = head?.next
        }
        var tail = head
        for _ in 0...b-a {
            tail = tail?.next
        }
        head?.next = list2
        var list2Tail = list2
        while list2Tail?.next != nil {
            list2Tail = list2Tail?.next
        }
        list2Tail?.next = tail?.next
        return dump.next
    }
    
    /// 题目链接：[2319. 判断矩阵是否是一个 X 矩阵](https://leetcode.cn/problems/check-if-matrix-is-x-matrix/)
    func checkXMatrix(_ grid: [[Int]]) -> Bool {
        let n = grid.count
        for i in 0..<n {
            for j in 0..<n {
                if (i == j || i + j == n - 1) && grid[i][j] == 0 {
                    return false
                } else if (i != j && i + j != n - 1) && grid[i][j] != 0 {
                    return false
                }
            }
        }
        return true
    }
    
    /// 题目链接：[2325. 解密消息](https://leetcode.cn/problems/decode-the-message/)
    func decodeMessage(_ key: String, _ message: String) -> String {
        let lowLetters: [Character] = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
        var map = [Character: Character](), curr = 0
        for char in key where char.isLowercase && !map.keys.contains(char) {
            map[char] = lowLetters[curr]
            curr += 1
        }
        return message.reduce("") { partialResult, char in
            if char.isLowercase { return "\(partialResult)\(map[char]!)" }
            return "\(partialResult)\(char)"
        }
    }
    
    /// 题目链接：[2558. 从数量最多的堆取走礼物](https://leetcode.cn/problems/take-gifts-from-the-richest-pile/)
    func pickGifts(_ gifts: [Int], _ k: Int) -> Int {
        var gifts = gifts.sorted()
        for _ in 0..<k {
            var i = -1, maxGift = -1
            for (j, gift) in gifts.enumerated() where gift > maxGift {
                i = j
                maxGift = gift
            }
            gifts[i] = Int(sqrt(Double(maxGift)))
        }
        return gifts.reduce(0, +)
    }
    
    /// 题目链接：[1604. 警告一小时内使用相同员工卡大于等于三次的人](https://leetcode.cn/problems/alert-using-same-key-card-three-or-more-times-in-a-one-hour-period/description/)
    func alertNames(_ keyName: [String], _ keyTime: [String]) -> [String] {
        let n = keyName.count, arr = zip(keyName, keyTime).sorted { $0.1 < $1.1 }
        var ans: Set<String> = []
        var map = [String: [String]]()
        for i in 0..<n where !ans.contains(arr[i].0) {
            map[arr[i].0, default: []].append(arr[i].1)
            let cnt = map[arr[i].0]!.count
            if cnt >= 3 {
                let time1 = getKeyTimeValue(map[arr[i].0]![cnt - 3])
                let time3 = getKeyTimeValue(map[arr[i].0]![cnt - 1])
                if time1 < time3 && time3 - time1 <= 60 { ans.insert(arr[i].0) }
            }
        }
        func getKeyTimeValue(_ time: String) -> Int {
            let arr = time.split(separator: ":")
            return Int(arr[0])! * 60 + Int(arr[1])!
        }
        return ans.sorted()
    }
    
    /// 题目链接：[1233. 删除子文件夹](https://leetcode.cn/problems/remove-sub-folders-from-the-filesystem/description/)
    func removeSubfolders(_ folder: [String]) -> [String] {
        let folder = folder.sorted(), n = folder.count
        var ans: [String] = [folder[0]]
        for i in 1..<n {
            var offset: String.Index?
            if ans.last!.count < folder[i].count { offset = folder[i].index(folder[i].startIndex, offsetBy: ans.last!.count) }
            if let offset = offset, folder[i].hasPrefix(ans.last!) && folder[i][offset] == "/" { }
            else { ans.append(folder[i]) }
        }
        return ans
    }
    
    /// 题目链接：[6354. 找出数组的串联值](https://leetcode.cn/problems/find-the-array-concatenation-value/description/)
    func findTheArrayConcVal(_ nums: [Int]) -> Int {
        var ans = 0
        var l = 0, r = nums.count - 1
        while l <= r {
            var curr = 0
            if l == r { curr = nums[l] }
            else { curr = Int("\(nums[l])\(nums[r])")! }
            ans += curr
            l += 1
            r -= 1
        }
        return ans
    }
    
    /// 题目链接：[1138. 字母板上的路径](https://leetcode.cn/problems/alphabet-board-path/description/)
    func alphabetBoardPath(_ target: String) -> String {
        let aV = Character("a").asciiValue!
        var ans = "", curr = (0, 0)
        for c in target {
            let target = (((Int(c.asciiValue! - aV)) / 5), (Int(c.asciiValue! - aV)) % 5)
            if c == "z" {
                ans.append(String(repeating: target.1 - curr.1 > 0 ? "R" : "L", count: abs(target.1 - curr.1)))
                ans.append(String(repeating: target.0 - curr.0 > 0 ? "D" : "U", count: abs(target.0 - curr.0)))
            } else {
                ans.append(String(repeating: target.0 - curr.0 > 0 ? "D" : "U", count: abs(target.0 - curr.0)))
                ans.append(String(repeating: target.1 - curr.1 > 0 ? "R" : "L", count: abs(target.1 - curr.1)))
            }
            curr = target
            ans.append("!")
        }
        return ans
    }
    
    /// 题目链接：[2335. 装满杯子需要的最短总时长](https://leetcode.cn/problems/minimum-amount-of-time-to-fill-cups/description/)
    func fillCups(_ amount: [Int]) -> Int {
        let amount = amount.sorted()
        if amount[2] > amount[1] + amount[0] { return amount[2] }
        return (amount[0] + amount[1] + amount[2] + 1) / 2
    }
    
    /// 题目链接：[2564. 子字符串异或查询](https://leetcode.cn/problems/substring-xor-queries/description/)
    func substringXorQueries(_ s: String, _ queries: [[Int]]) -> [[Int]] {
        let queries = queries.map { $0[0] ^ $0[1] }, chars = [Character](s), n = s.count
        var ans = [[Int]](), map = [Int: [Int]]()
        for i in 0..<n {
            var curr = 0
            for j in i..<min(i+30, n) {
                curr = (curr << 1 + (chars[j] == "0" ? 0 : 1))
                if !map.keys.contains(curr) || j - i < map[curr]![1] - map[curr]![0] { map[curr] = [i, j] }
            }
        }
        for query in queries {
            ans.append(map[query, default: [-1, -1]])
        }
        return ans
    }
    
    /// 题目链接：[2341. 数组能形成多少数对](https://leetcode.cn/problems/maximum-number-of-pairs-in-array/description/)
    func numberOfPairs(_ nums: [Int]) -> [Int] {
        var map = [Int: Int]()
        nums.forEach { map[$0, default: 0] += 1 }
        var pairCnt = 0, remainder = 0
        for item in map {
            pairCnt += item.value / 2
            remainder += item.value % 2
        }
        return [pairCnt, remainder]
    }

    /// 题目链接：[2348. 全 0 子数组的数目](https://leetcode.cn/problems/number-of-zero-filled-subarrays/description/)
    func zeroFilledSubarray(_ nums: [Int]) -> Int {
        var cnt = 0, ans = 0
        for num in nums {
            if num == 0 {
                ans += (1 + cnt)
                cnt += 1
            } else {
                cnt = 0
            }
        }
        return ans
    }

    /// 题目链接：[2347. 最好的扑克手牌](https://leetcode.cn/problems/best-poker-hand/)
    func bestHand(_ ranks: [Int], _ suits: [Character]) -> String {
        if suits.allSatisfy({ $0 == suits[0] }) { return "Flush" }
        var cnts = [Int: Int]()
        ranks.forEach { cnts[$0, default: 0] += 1 }
        let cntMax = cnts.max { $0.value > $1.value }!.value
        if cntMax >= 3 { return "Three of a Kind" }
        if cntMax == 2 { return "Pair" }
        if cntMax == 1 { return "High Card" }
        return ""
    }

    /// 题目链接：[2357. 使数组中所有元素都等于零](https://leetcode.cn/problems/make-array-zero-by-subtracting-equal-amounts/description/)
    func minimumOperations(_ nums: [Int]) -> Int {
        return Set<Int>(nums).filter { $0 > 0 }.count
    }
    
    func mergeArrays(_ nums1: [[Int]], _ nums2: [[Int]]) -> [[Int]] {
        let n = nums1.count, m = nums2.count
        var l1 = 0, l2 = 0
        var ans = [[Int]]()
        while l1 < n && l2 < m {
            if nums1[l1][0] == nums2[l2][0] {
                ans.append([nums1[l1][0], nums1[l1][1] + nums2[l2][1]])
                l1 += 1
                l2 += 1
            } else if nums1[l1][0] < nums2[l2][0] {
                ans.append([nums1[l1][0], nums1[l1][1]])
                l1 += 1
            } else {
                ans.append([nums2[l2][0], nums2[l2][1]])
                l2 += 1
            }
        }
        while l1 < n {
            ans.append([nums1[l1][0], nums1[l1][1]])
            l1 += 1
        }
        while l2 < m {
            ans.append([nums2[l2][0], nums2[l2][1]])
            l2 += 1
        }
        return ans
    }
    
    func minOperations(_ n: Int) -> Int {
        var n = n
        var ans = 0
        while n != 0 {
            if n.nonzeroBitCount > (n + lowBit(x: n)).nonzeroBitCount {
                n += lowBit(x: n)
            } else {
                n -= lowBit(x: n)
            }
            ans += 1
        }
        func lowBit(x: Int) -> Int {
            return x & -x
        }
        return ans
    }
    
    func squareFreeSubsets(_ nums: [Int]) -> Int {
        let nums = nums.filter { !($0 % 4 == 0 || $0 % 9 == 0 || $0 % 16 == 0 || $0 % 25 == 0) }.sorted()
        let n = nums.count, MOD = Int(1e9 + 7)
        var dp = [Int](repeating: 1, count: n)
        dp[0] = 1
        for i in 1..<n {
            for j in 0..<i where !checkSim(a: nums[i], b: nums[j]) {
                dp[i] += dp[j] + 1
            }
        }
        func checkSim(a: Int, b: Int) -> Bool {
            if a == b || b % a == 0 || a % b == 0 { return true }
            if a % 2 == 0 && b % 2 == 0 { return true }
            if a % 3 == 0 && b % 3 == 0 { return true }
            if a % 5 == 0 && b % 5 == 0 { return true }
            return false
        }
        return dp[n - 1]
    }
    
    func braceExpansionII(_ expression: String) -> [String] {
        let expression = [Character](expression), n = expression.count
        var op = [Character]()
        var stack = [Set<String>]()
        for i in 0..<n {
            if expression[i] == "," {
                /// 不断的弹出栈顶运算符，直到栈顶为空或者栈顶不为 *
                while !op.isEmpty && op.last! == "*" {
                    ope()
                }
                op.append("+")
            } else if expression[i] == "{" {
                if i > 0 && (expression[i - 1] == "}" || expression[i - 1].isLetter) {
                    op.append("*")
                }
                op.append("{")
            } else if expression[i] == "}" {
                while !op.isEmpty && op.last != "{" {
                    ope()
                }
                op.removeLast()
            } else {
                if i > 0 && (expression[i - 1] == "}" || expression[i - 1].isLetter) {
                    op.append("*")
                }
                var str = "\(expression[i])"
                stack.append(Set<String>([str]))
            }
        }
        while !op.isEmpty {
            ope()
        }
        return stack.last!.sorted()
        
        func ope() {
            let l = stack.count - 2, r = stack.count - 1
            if op.last! == "+" {
                for right in stack[r] {
                    stack[l].insert(right)
                }
            } else {
                var tmp = Set<String>()
                for left in stack[l] {
                    for right in stack[r] {
                        tmp.insert(left + right)
                    }
                }
                stack[l] = tmp
            }
            op.removeLast()
            stack.removeLast()
        }
    }
    
    func minimumRecolors(_ blocks: String, _ k: Int) -> Int {
        let n = blocks.count, chars = [Character](blocks)
        var l = 0, r = 0
        var ans = Int.max, wCnt = 0
        while r < n {
            wCnt += chars[r] == "W" ? 1 : 0
            if r - l == k {
                wCnt -= chars[l] == "W" ? 1 : 0
                l += 1
            }
            if r - l == k - 1 {
                ans = min(ans, wCnt)
            }
            r += 1
        }
        return ans
    }
    
    func minSubarray(_ nums: [Int], _ p: Int) -> Int {
        let n = nums.count
        var s = [Int](repeating: 0, count: n + 1)
        for (i, num) in nums.enumerated() {
            s[i + 1] = (s[i] + num) % p
        }
        let x = s[n]
        //        if x == 0 { return 0 }
        var ans = n
        var last = [Int: Int]()
        for i in 0...n {
            last[s[i]] = i
            let j = last[(s[i] - x + p) % p, default: -n]
            ans = min(ans, i - j)
        }
        return ans >= n ? -1 : ans
    }
    
    func mergeStones(_ stones: [Int], _ k: Int) -> Int {
        let n = stones.count
        if (n - 1) % (k - 1) > 0 {
            return -1
        }
        var preSum = [Int](repeating: 0, count: n + 1)
        for i in 0..<n {
            preSum[i + 1] = preSum[i] + stones[i]
        }
        var memo = [[[Int]]](repeating: [[Int]](repeating: [Int](repeating: -1, count: k + 1), count: n), count: n)
        func dfs(_ i: Int, _ j: Int, _ p: Int) -> Int {
            if memo[i][j][p] != -1 {
                return memo[i][j][p]
            }
            var res = Int.max
            if p == 1 {
                if i == j {
                    res = 0
                } else {
                    res = (dfs(i, j, k) + preSum[j + 1] - preSum[i])
                }
            } else {
                var m = i
                while m < j {
                    res = min(res, dfs(i, m, 1) + dfs(m + 1, j, p - 1))
                    m += k - 1
                }
            }
            memo[i][j][p] = res
            return res
        }
        return dfs(0, n - 1, 1)
    }
    
    func largestValsFromLabels(_ values: [Int], _ labels: [Int], _ numWanted: Int, _ useLimit: Int) -> Int {
        let n = values.count
        var ans = 0, cnt = 0
        var arr = zip(values, labels).sorted { $0.0 > $1.0 }
        var map = [Int: Int]()
        for item in arr {
            if cnt >= numWanted { break }
            if map[item.1, default: 0] >= useLimit { continue }
            ans += item.0
            map[item.1, default: 0] += 1
            cnt += 1
        }
        return ans
    }
    
    override var excuteable: Bool { return true }
    
    override func executeTestCode() {
        super.executeTestCode()
        print(largestValsFromLabels([5,4,3,2,1], [1,3,3,3,2], 3, 2))
    }
}

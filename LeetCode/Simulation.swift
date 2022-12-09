//
//  Simulation.swift
//  LeetCode
//
//  Created by haodong xu on 2022/9/4.
//

import Foundation

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

//    override var excuteable: Bool { return true }

    override func executeTestCode() {
        super.executeTestCode()
        print(numDifferentIntegers("leet1234code234"))
    }
}

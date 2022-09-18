//
//  main.swift
//  LeetCode
//
//  Created by haodong xu on 2021/12/29.
//

import Foundation
import CoreImage

class Solution {
    func rotate(_ nums: inout [Int], _ k: Int) {
        func reverse(_ nums: inout [Int], start: Int, end: Int) {
            var start = start, end = end
            while start < end {
                let temp = nums[start]
                nums[start] = nums[end]
                nums[end] = temp
                start += 1
                end -= 1
            }
        }
        let n = nums.count
        let k = k % n
        reverse(&nums, start: 0, end: n - 1)
        reverse(&nums, start: 0, end: k - 1)
        reverse(&nums, start: k, end: n - 1)
    }

    func lowestCommonAncestor(_ root: TreeNode?, _ p: TreeNode?, _ q: TreeNode?) -> TreeNode? {
        guard let root = root else { return nil }
        if root === p || root === q {
            return root
        }
        let left = lowestCommonAncestor(root.left, p, q)
        let right = lowestCommonAncestor(root.right, p, q)
        if left == nil && right == nil {
            return nil
        }
        if left == nil {
            return right
        }
        if right == nil {
            return left
        }
        return root
    }

    func reverseList(_ head: ListNode?) -> ListNode? {
        var pre: ListNode?
        var curr = head
        while curr != nil {
            let temp = curr?.next
            curr?.next = pre
            pre = curr
            curr = temp
        }
        return pre
    }

    func isPalindrome(_ head: ListNode?) -> Bool {
        var fast = head, slow = head
        var curr = head, pre: ListNode?
        while fast != nil && fast?.next != nil {
            curr = slow
            slow = slow?.next
            fast = fast?.next?.next
            curr?.next = pre
            pre = curr
        }
        if fast != nil {
            slow = slow?.next
        }
        while curr != nil && slow != nil {
            if curr!.val != slow!.val {
                return false
            }
            curr = curr?.next
            slow = slow?.next
        }
        return true
    }

    func longestCommonSubsequence(_ text1: String, _ text2: String) -> Int {
        let n = text1.count, m = text2.count
        var dp = [[Int]](repeating: [Int](repeating: 0, count: m + 1), count: n + 1)
        var chars1 = [Character](text1), chars2 = [Character](text2)
        for i in 1...n {
            for j in 1...m {
                if chars1[i - 1] == chars2[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1] + 1
                } else {
                    dp[i][j] = max(dp[i][j - 1], dp[i - 1][j])
                }
            }
        }
        return dp[n][m]
    }

    func hasCycle(_ head: ListNode?) -> Bool {
        guard let head = head else { return false }
        var slow:ListNode? = head, fast = head.next
        while slow !== fast {
            if fast == nil {
                return false
            }
            fast = fast?.next?.next
            slow = slow?.next
        }
        return true
    }

    func detectCycle(_ head: ListNode?) -> ListNode? {
        guard let head = head else { return nil }
        var slow: ListNode? = head, fast: ListNode? = head
        repeat {
            if fast == nil || fast?.next == nil {
                return nil
            }
            fast = fast?.next?.next
            slow = slow?.next
        } while slow !== fast
        fast = head
        while fast !== slow {
            fast = fast?.next
            slow = slow?.next
        }
        return slow
    }

    func lengthOfLIS(_ nums: [Int]) -> Int {
        let n = nums.count
        var dp = [Int](repeating: 1, count: n)
        var ans = -1
        for i in 0..<n {
            for j in 0..<i {
                if nums[j] < nums[i] {
                    dp[i] = max(dp[i], dp[j] + 1)
                }
            }
            ans = max(ans, dp[i])
        }
        return ans
    }

    func solveEquation(_ equation: String) -> String {
        var x = 0, num = 0, op = 1
        let chars = [Character](equation)
        let n = chars.count
        var i = 0
        while i < n {
            if chars[i] == "+" {
                op = 1
                i += 1
            } else if chars[i] == "-" {
                op = -1
                i += 1
            } else if chars[i] == "=" {
                x *= -1
                num *= -1
                op = 1
                i += 1
            } else {
                var j = i, str = ""
                while j < n && (chars[j] != "+" && chars[j] != "-" && chars[j] != "=") {
                    str.append(chars[j])
                    j += 1
                }
                if str.last == "x" {
                    str.removeLast()
                    x += (str.count == 0 ? 1 * op : Int(str)! * op)
                } else {
                    num += (Int(str)! * op)
                }
                i = j
            }
        }
        if x == 0 {
            return num == 0 ? "Infinite solutions" : "No solution"
        }
        return "x=\(num / -x)"
    }

    func reformat(_ s: String) -> String {
        var nums = [Character](), letters = [Character]()
        for char in s {
            if char.isNumber {
                nums.append(char)
            } else {
                letters.append(char)
            }
        }
        if abs(nums.count - letters.count) > 1 { return "" }
        var ans = ""
        for i in 0..<(min(nums.count, letters.count)) {
            ans.append(nums[i])
            ans.append(letters[i])
        }
        if nums.count > letters.count {
            ans.append(nums.last!)
        } else if nums.count < letters.count {
            ans = "\(letters.last!)" + ans
        }
        return ans
    }

    func groupThePeople(_ groupSizes: [Int]) -> [[Int]] {
        var map = [Int: [Int]]()
        var ans = [[Int]]()
        for (i, groupSize) in groupSizes.enumerated() {
            if let _ = map[groupSize] {
                map[groupSize]?.append(i)
            } else {
                map[groupSize] = [i]
            }
        }
        for (key, value) in map {
            var arr = [Int]()
            for (i, num) in value.enumerated() {
                arr.append(num)
                if (i + 1) % key == 0 {
                    ans.append(arr)
                    arr.removeAll()
                }
            }
        }
        return ans
    }

    func fourSumCount(_ nums1: [Int], _ nums2: [Int], _ nums3: [Int], _ nums4: [Int]) -> Int {
        var map = [Int: Int]()
        var ans = 0
        for num1 in nums1 {
            for num2 in nums2 {
                map[num1 + num2, default: 0] += 1
            }
        }
        for num3 in nums3 {
            for num4 in nums4 {
                if let cnt = map[-num3 - num4] {
                    ans += cnt
                }
            }
        }
        return ans
    }

    func longestSubstring(_ s: String, _ k: Int) -> Int {
        var cnts = [Int](repeating: 0, count: 26)
        let chars = [Character](s)
        let aCharV = Int(Character("a").asciiValue!)
        let n = s.count
        var ans = 0
        for i in 1...26 {
            for j in 0..<26 {
                cnts[j] = 0
            }
            var currTotal = 0
            var sum = 0
            var j = 0
            for l in 0..<n {
                let charIndex = Int(chars[l].asciiValue!) - aCharV
                cnts[charIndex] += 1
                if cnts[charIndex] == 1 {
                    currTotal += 1
                }
                if cnts[charIndex] == k {
                    sum += 1
                }
                while currTotal > i {
                    let charIndex = Int(chars[j].asciiValue!) - aCharV
                    cnts[charIndex] -= 1
                    if cnts[charIndex] == 0 {
                        currTotal -= 1
                    }
                    if cnts[charIndex] == k - 1 {
                        sum -= 1
                    }
                    j += 1
                }
                if currTotal == sum {
                    ans = max(ans, l - j + 1)
                }
            }
        }
        return ans
    }

    func getSum(_ a: Int, _ b: Int) -> Int {
        return b == 0 ? a : getSum(a ^ b, (a & b) << 1)
    }

    func topKFrequent(_ nums: [Int], _ k: Int) -> [Int] {
        var map = [Int: Int]()
        for num in nums {
            map[num, default: 0] += 1
        }
        var arr = [(Int, Int)]()
        for item in map {
            arr.append((item.value, item.key))
        }
        func findKth(l: Int, r: Int) {
            if l >= r {
                return
            }
            let random = Int.random(in: l...r)
            let th = qsortBy(kth: random, l: l, r: r)
            if th == arr.count - k {
                return
            } else if th < arr.count - k {
                findKth(l: th + 1, r: r)
            } else {
                findKth(l: l, r: th)
            }
        }
        func qsortBy(kth: Int, l: Int, r: Int) -> Int {
            swap(i: kth, j: l)
            var j = l + 1
            for i in l+1...r{
                if arr[i].0 < arr[l].0 {
                    swap(i: i, j: j)
                    j += 1
                }
            }
            swap(i: l, j: j - 1)
            return j - 1
        }
        func swap(i: Int, j: Int) {
            let temp = arr[i]
            arr[i] = arr[j]
            arr[j] = temp
        }
        findKth(l: 0, r: arr.count - 1)
        var ans = [Int]()
        for i in arr.count-k..<arr.count {
            ans.append(arr[i].1)
        }
        return ans
    }

    func increasingTriplet(_ nums: [Int]) -> Bool {
        guard nums.count >= 3 else { return false }
        var first = nums[0], second = Int.max
        for num in nums {
            if num > second {
                return true
            } else if num > first {
                second = num
            } else {
                first = num
            }
        }
        return false
    }

    func oddEvenList(_ head: ListNode?) -> ListNode? {
        let evenDump: ListNode? = ListNode.init(-1)
        var evenCurr = evenDump
        var odd = head
        while odd?.next != nil {
            let even = odd?.next
            evenCurr?.next = even
            evenCurr = evenCurr?.next
            if let node = odd?.next?.next {
                odd?.next = node
                odd = odd?.next
            } else {
                break
            }
        }
        evenCurr?.next = nil
        odd?.next = evenDump?.next
        return head
    }

    func coinChange(_ coins: [Int], _ amount: Int) -> Int {
        if amount == 0 {
            return 0
        }
        var memo = [Int].init(repeating: -1, count: amount + 1)
        func dfs(amount: Int) -> Int {
            if amount < 0 {
                return -1
            } else if amount == 0 {
                return 0
            }
            if memo[amount] != 0 {
                return memo[amount - 1]
            }
            var minV = Int.max
            for coin in coins where amount >= coin {
                let curr = dfs(amount: amount - coin)
                if curr != -1 {
                    minV = min(minV, curr + 1)
                }
            }
            memo[amount] = minV == Int.max ? -1 : minV
            return memo[amount]
        }
        return dfs(amount: amount)
    }

    func numSquares(_ n: Int) -> Int {
        var dp = [Int](repeating: 0, count: n + 1)
        for i in 1...n {
            var j = 1
            var minV = Int.max
            while j * j <= i {
                minV = min(minV, dp[i - j * j])
                j += 1
            }
            dp[i] = minV + 1
        }
        return dp[n]
    }

    func productExceptSelf(_ nums: [Int]) -> [Int] {
        let n = nums.count
        var left = [Int](repeating: 1, count: n + 1)
        var right = [Int](repeating: 1, count: n + 1)
        var ans = [Int](repeating: 0, count: n)
        for i in 1...n {
            left[i] = left[i - 1] * nums[i - 1]
        }
        for i in (0..<n).reversed() {
            right[i] = right[i + 1] * nums[i]
        }
        for i in 0..<n {
            ans[i] = left[i] * right[i + 1]
        }
        return ans
    }

    func kthSmallest(_ root: TreeNode?, _ k: Int) -> Int {
        var ans = -1, k = k
        func dfs(node: TreeNode?) {
            guard let node = node else {
                return
            }
            if k < 0 {
                return
            }
            dfs(node: node.left)
            k -= 1
            if k == 0 {
                ans = node.val
            }
            dfs(node: node.right)
        }
        dfs(node: root)
        return ans
    }

    func findKthLargest(_ nums: [Int], _ k: Int) -> Int {
        var nums = nums
        let n = nums.count
        func qsort(l: Int, r: Int) {
            if l >= r {
                return
            }
            let random = Int.random(in: l...r)
            let th = qsortBy(curr: random, l: l, r: r)
            if th == n - k {
                return
            } else if th < n - k {
                qsort(l: th + 1, r: r)
            } else {
                qsort(l: l, r: th)
            }
        }
        func qsortBy(curr: Int, l: Int, r: Int) -> Int {
            nums.swapAt(curr, l)
            var j = l + 1
            for i in l+1...r {
                if nums[i] < nums[l] {
                    nums.swapAt(i, j)
                    j += 1
                }
            }
            nums.swapAt(l, j - 1)
            return j - 1
        }
        qsort(l: 0, r: n - 1)
        return nums[n - k]
    }

    func maxScore(_ s: String) -> Int {
        let n = s.count, chars = [Character](s)
        var left = [Int](repeating: 0, count: n + 1)
        var right = [Int](repeating: 0, count: n + 1)
        for i in 1...n {
            let curr = chars[i - 1] == "0" ? 1 : 0
            left[i] = left[i - 1] + curr
        }
        for i in (0..<n).reversed() {
            let curr = chars[i] == "1" ? 1 : 0
            right[i] = right[i + 1] + curr
        }
        var ans = 0
        for i in 1..<n {
            ans = max(ans, left[i] + right[i])
        }
        return ans
    }

    func copyRandomList(_ head: SingalNode?) -> SingalNode? {
        guard let head = head else { return nil }
        var node: SingalNode? = head
        var map = [SingalNode?: SingalNode]()
        while let tempNode = node {
            map[tempNode] = SingalNode(tempNode.val)
            node = node?.next
        }
        node = head
        let ans = map[head]
        var curr = ans
        while let tempNode = node {
            curr?.next = map[tempNode.next]
            curr?.random = map[tempNode.random]
            node = node?.next
            curr = curr?.next
        }
        return ans
    }

    func multiply(_ num1: String, _ num2: String) -> String {
        if num1 == "0" || num2 == "0" { return "0" }
        var base1 = 1
        var ans = 0
        let asciiValue_0 = Character("0").asciiValue!
        for char1 in num1.reversed() {
            var base2 = 1
            for char2 in num2.reversed() {
                let n1 = char1.asciiValue! - asciiValue_0
                let n2 = char2.asciiValue! - asciiValue_0
                ans += Int(n1 * n2) * base1 * base2
                base2 *= 10
            }
            base1 *= 10
        }
        return "\(ans)"
    }

    func deepestLeavesSum(_ root: TreeNode?) -> Int {
        guard let root = root else { return -1 }
        var queue: [TreeNode] = [root]
        var ans = 0
        while !queue.isEmpty {
            let n = queue.count
            ans = 0
            for _ in 0..<n {
                let node = queue.removeFirst()
                ans += node.val
                if let left = node.left {
                    queue.append(left)
                }
                if let right = node.right {
                    queue.append(right)
                }
            }
        }
        return ans
    }

    func countSubstrings(_ s: String) -> Int {
        let n = s.count, chars = [Character](s)
        var dp = [[Bool]](repeating: [Bool](repeating: false, count: n), count: n)
        var ans = 0
        for i in 0..<n {
            for j in 0...i {
                if chars[i] == chars[j] && (i - j < 2 || dp[i - 1][j + 1]) {
                    dp[i][j] = true
                    ans += 1
                }
            }
        }
        return ans
    }

    func flatten(_ root: TreeNode?) {
        var curr = root
        while curr != nil {
            if curr?.left != nil {
                let next = curr?.left
                var predecessor = next
                while predecessor?.right != nil {
                    predecessor = predecessor?.right
                }
                predecessor?.right = curr?.right
                curr?.left = nil
                curr?.right = next
            }
            curr = curr?.right
        }
    }

    func nextPermutation(_ nums: inout [Int]) {
        let n = nums.count
        var i = n - 2
        while i >= 0 && nums[i] >= nums[i + 1] {
            i -= 1
        }
        if i >= 0 {
            var j = n - 1
            while j >= 0 && nums[i] >= nums[j] {
                j -= 1
            }
            nums.swapAt(i, j)
        }
        func reverse(_ start: Int,_ end: Int) {
            var start = start, end = end
            while start < end {
                nums.swapAt(start, end)
                start += 1
                end -= 1
            }
        }
        reverse(i + 1, n - 1)
    }

    func maxEqualFreq(_ nums: [Int]) -> Int {
        var cnts = [Int: Int](), freq = [Int: Int]()
        var maxSum = 0
        var ans = 0
        for (index, num) in nums.enumerated() {
            let len = index + 1
            cnts[num, default: 0] += 1
            freq[cnts[num]!, default: 0] += 1
            freq[cnts[num]! - 1, default: 0] -= 1
            maxSum = max(maxSum, cnts[num]!)
            if maxSum == 1 {
                ans = len
            } else if freq[maxSum, default: 0] * maxSum + 1 == len {
                ans = len
            } else if (maxSum - 1) * (freq[maxSum - 1, default: 0] + 1) + 1 == len {
                ans = len
            }
        }
        return ans
    }

    func longestConsecutive(_ nums: [Int]) -> Int {
        var map = [Int: Int](), ans = 0
        for num in nums {
            if map[num] != nil { continue }
            let left = map[num - 1, default: 0]
            let right = map[num + 1, default: 0]
            map[num] = left + right + 1
            if left != 0 {
                map[num - left] = left + right + 1
            }
            if right != 0 {
                map[num + right] = left + right + 1
            }
            ans = max(ans, left + right + 1)
        }
        return ans
    }

    func busyStudent(_ startTime: [Int], _ endTime: [Int], _ queryTime: Int) -> Int {
        let n = startTime.count
        var ans = 0
        for i in 0..<n {
            if startTime[i] <= queryTime && queryTime >= endTime[i] {
                ans += 1
            }
        }
        return ans
    }

    func numSimilarGroups(_ strs: [String]) -> Int {
        let n = strs.count
        var parents = [Int](repeating: 0, count: n)
        for i in 0..<n {
            parents[i] = i
        }
        for i in 0..<n-1 {
            for j in i+1..<n {
                let rootI = find(i), rootJ = find(j)
                if rootI == rootJ {
                    continue
                }
                if check(strs[i], strs[j]) {
                    union(i, j)
                }
            }
        }
        func union(_ i: Int, _ j: Int) {
            let rootI = find(i)
            let rootJ = find(j)
            if rootI != rootJ {
                parents[rootI] = rootJ
            }
        }
        func find(_ i: Int) -> Int {
            var i = i
            while parents[i] != i {
                i = parents[i]
            }
            return i
        }
        func check(_ str1: String, _ str2: String) -> Bool {
            if str1.count != str2.count {
                return false
            }
            var i1 = str1.startIndex, i2 = str2.startIndex
            var diff = 0
            while i1 != str1.endIndex && i2 != str2.endIndex {
                if str1[i1] != str2[i2] {
                    diff += 1
                }
                if diff > 2 {
                    return false
                }
                i1 = str1.index(after: i1)
                i2 = str2.index(after: i2)
            }
            return true
        }
        var ans = 0
        for i in 0..<n where parents[i] == i {
            ans += 1
        }
        return ans
    }

    func findCircleNum(_ isConnected: [[Int]]) -> Int {
        let n = isConnected.count
        var parents = [Int](repeating: 0, count: n)
        for i in 0..<n {
            parents[i] = i
        }
        for i in 0..<n {
            for j in 0..<n {
                let rootI = find(i), rootJ = find(j)
                if rootI == rootJ {
                    continue
                }
                if isConnected[i][j] == 1 {
                    union(i, j)
                }
            }
        }
        func union(_ i: Int, _ j: Int) {
            parents[find(i)] = find(j)
        }
        func find(_ i: Int) -> Int {
            var i = i
            while parents[i] != i {
                i = parents[i]
            }
            return i
        }
        var ans = 0
        for i in 0..<n where parents[i] == i {
            ans += 1
        }
        return ans
    }

    func minFlipsMonoIncr(_ s: String) -> Int {
        let n = s.count
        var dp = [[Int]](repeating: [0, 0], count: n + 1)
        var si = s.startIndex, i = 1
        while si != s.endIndex {
            dp[i][0] = dp[i - 1][0] + (s[si] == "1" ? 1 : 0)
            dp[i][1] = min(dp[i - 1][0], dp[i - 1][1]) + (s[si] == "0" ? 1 : 0)
            si = s.index(after: si)
            i += 1
        }
        return min(dp[n][0], dp[n][1])
    }

    func minCost(_ costs: [[Int]]) -> Int {
        let n = costs.count
        var dp = [[Int]](repeating: [0, 0, 0], count: n + 1)
        for i in 1...n {
            dp[i][0] = min(dp[i - 1][1], dp[i - 1][2]) + costs[i - 1][0]
            dp[i][1] = min(dp[i - 1][0], dp[i - 1][2]) + costs[i - 1][1]
            dp[i][2] = min(dp[i - 1][1], dp[i - 1][0]) + costs[i - 1][2]
        }
        return min(dp[n][0], dp[n][1], dp[n][2])
    }

    func isPrefixOfWord(_ sentence: String, _ searchWord: String) -> Int {
        let words = sentence.split(separator: " ")
        for (i, word) in words.enumerated() {
            if word.starts(with: searchWord) {
                return i + 1
            }
        }
        return -1
    }

    func constructMaximumBinaryTree(_ nums: [Int]) -> TreeNode? {
        guard  nums.count > 0 else { return nil }
        var maxI = 0, maxV = nums[0]
        let n = nums.count
        for i in 0..<n where nums[i] > maxV {
            maxI = i
            maxV = nums[i]
        }
        let ans = TreeNode(maxV)
        if maxI != 0 {
            var left = [Int]()
            for i in 0..<maxI {
                left.append(nums[i])
            }
            ans.left = constructMaximumBinaryTree(left)
        }
        if maxI != n - 1 {
            var right = [Int]()
            for i in maxI+1..<n {
                right.append(nums[i])
            }
            ans.right = constructMaximumBinaryTree(right)
        }
        return ans
    }

    func printTree(_ root: TreeNode?) -> [[String]] {
        func getHeight(node: TreeNode?) -> Int {
            guard let node = node else { return 0 }
            let left = getHeight(node: node.left)
            let right = getHeight(node: node.right)
            return max(left, right) + 1
        }
        let height = getHeight(node: root) - 1
        let m = height + 1, n = Int(pow(2.0, Double(height + 1))) - 1
        var ans = [[String]](repeating: [String](repeating: "", count: n), count: m)
        func fillNode(node: TreeNode?, i: Int, j: Int) {
            guard let node = node else { return }
            ans[i][j] = "\(node.val)"
            let offsetX = Int(Double(pow(2.0, Double(height - i - 1))))
            fillNode(node: node.left, i: i + 1, j: j - offsetX)
            fillNode(node: node.right, i: i + 1, j: j + offsetX)
        }
        fillNode(node: root, i: 0, j: (n - 1) / 2)
        return ans
    }

    func longestIncreasingPath(_ matrix: [[Int]]) -> Int {
        let n = matrix.count, m = matrix[0].count
        var memo = [[Int]](repeating: [Int](repeating: -1, count: m), count: n)
        let dirs = [[-1, 0], [0, -1], [1, 0], [0, 1]]
        var ans = 0
        func dfs(i: Int, j: Int) -> Int {
            if memo[i][j] != -1 {
                return memo[i][j]
            }
            var maxV = 0
            for dir in dirs {
                let nextI = dir[0] + i
                let nextJ = dir[1] + j
                if nextI >= 0 && nextI < n && nextJ >= 0 && nextJ < m && matrix[nextI][nextJ] > matrix[i][j] {
                    maxV = max(maxV, dfs(i: nextI, j: nextJ))
                }
            }
            memo[i][j] = maxV + 1
            return memo[i][j]
        }
        for i in 0..<n {
            for j in 0..<m {
                ans = max(dfs(i: i, j: j), ans)
            }
        }
        return ans
    }

    func ladderLength(_ beginWord: String, _ endWord: String, _ wordList: [String]) -> Int {
        guard beginWord != endWord && wordList.contains(endWord) else { return 0 }
        let n = wordList.count
        var visvited = [Bool](repeating: false, count: n)
        var queue: [String] = [beginWord]
        var height = 0
        func isShortTranslateList(_ str1: String,_ str2: String) -> Bool {
            if str1.count != str2.count { return false }
            var diff = 0
            var i1 = str1.startIndex, i2 = str2.startIndex
            while i1 != str1.endIndex {
                if str1[i1] != str2[i2] {
                    diff += 1
                }
                if diff >= 2 {
                    return false
                }
                i1 = str1.index(after: i1)
                i2 = str2.index(after: i2)
            }
            return true
        }
        while !queue.isEmpty {
            let cnt = queue.count
            for _ in 0..<cnt {
                let curr = queue.removeFirst()
                if curr == endWord {
                    return height + 1
                }
                for (j, word) in wordList.enumerated() where !visvited[j] && isShortTranslateList(curr, word) {
                    queue.append(word)
                    visvited[j] = true
                }
            }
            height += 1
        }
        return 0
    }

    func canBeEqual(_ target: [Int], _ arr: [Int]) -> Bool {
        var map = [Int: Int]()
        for num in target {
            map[num, default: 0] += 1
        }
        for num in arr {
            map[num, default: 0] -= 1
            if map[num]! < 0 {
                return false
            }
        }
        return true
    }

    func findClosestElements(_ arr: [Int], _ k: Int, _ x: Int) -> [Int] {
        let n = arr.count
        var l = 0, r = n - 1
        while l < r {
            let mid = (l + r) >> 1
            if arr[mid] >= x {
                r = mid
            } else {
                l = mid + 1
            }
        }
        l = r - 1;
        var curr = k
        while curr > 0 {
            if l < 0 {
                r += 1
            } else if r >= n {
                l -= 1
            } else if x - arr[l] <= arr[r] - x {
                l -= 1
            } else {
                r += 1
            }
            curr -= 1
        }
        var ans = [Int]()
        for i in l+1..<r {
            ans.append(arr[i])
        }
        return ans
    }
}

let baseCodes = registerBaseCode()
for baseCode in baseCodes where baseCode.excuteable {
    baseCode.executeTestCode()
}

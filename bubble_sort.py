def bubble_sort(arr):
    """
    Bubble Sort Algorithm
    Time Complexity: O(n^2)
    Space Complexity: O(1)
    """
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
    return arr


if __name__ == "__main__":
    test_array = [64, 34, 25, 12, 22, 11, 90]
    print("Original array:", test_array)
    sorted_array = bubble_sort(test_array.copy())
    print("Sorted array:", sorted_array)

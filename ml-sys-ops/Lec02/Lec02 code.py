import time
import multiprocessing

def is_prime(n):
    """Checks if a number is prime."""
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def sequential_prime_check(numbers):
    return [is_prime(n) for n in numbers]

def parallel_prime_check(numbers):
    with multiprocessing.Pool() as pool:
        return pool.map(is_prime, numbers)

def main(min, max):
    numbers_to_check = list(range(min, max))
    print(f"\nChecking primality for numbers from {min} to {max - 1}")
    # Sequential
    start_time = time.time()
    sequential_results = sequential_prime_check(numbers_to_check)
    end_time = time.time()
    print(f"Sequential execution time: {end_time - start_time:.4f} seconds")

    # Parallel
    start_time = time.time()
    parallel_results = parallel_prime_check(numbers_to_check)
    end_time = time.time()
    print(f"Parallel execution time: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    num_cores = multiprocessing.cpu_count()
    print(f"\n\n\nNumber of available CPU cores: {num_cores}")
    min = 10_000_000
    max = 10_000_501
    main(min, max) # small set of numbers
    min = 10_000_000
    max = 20_000_001
    main(min, max) # large set of numbers
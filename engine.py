import argparse

def main():
  parser = argparse.ArgumentParser(description='A simple command line script.')
  parser.add_argument('-n', '--name', type=str, required=True, help='Name of the person')
  parser.add_argument('-a', '--age', type=int, required=False, help='Age of the person')
  
  args = parser.parse_args()

  print(f"Hello, {args.name}!")
  if args.age:
    print(f"You are {args.age} years old.")

if __name__ == '__main__':
  main()

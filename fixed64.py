while True:
  num = int(input("Enter input num: ")).to_bytes(8, byteorder="big", signed=True)
  i_part = int.from_bytes(num[:4], byteorder="big", signed=True)
  f_part = int.from_bytes(num[4:], byteorder="big", signed=False)/0x100000000
  print(f"I:{i_part} F:{f_part}")

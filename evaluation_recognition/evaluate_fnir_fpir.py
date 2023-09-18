import os

def calculate_fpir(result):
  list_file = os.listdir(result)

  len_file = 0
  unknown = 0

  for file in list_file:
      file_path = os.path.join(result, file)
      list_name = os.listdir(file_path)
      for name in list_name:
          len_file += 1
          name_path = os.path.join(file_path, name)
          with open(name_path, "r") as f:
              list_pred = f.readlines()

          for pred in list_pred:
              if pred == "Unknown":
                  unknown += 1
                  # print(name)

  return (len_file - unknown)/len_file

def calculate_fnir(result):
  list_file = os.listdir(result)

  fni = 0
  len_file = 0

  for file in list_file:
      file_path = os.path.join(result, file)
      list_name = os.listdir(file_path)
      for name in list_name:
          len_file += 1
          name_path = os.path.join(file_path, name)
          with open(name_path, "r") as f:
              list_pred = f.readlines()
          if name[:-9] not in list_pred:
              fni += 1
  return fni/len_file

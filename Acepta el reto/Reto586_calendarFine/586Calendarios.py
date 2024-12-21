interacciones = int(input())
contador=0
salidad_data=""
while contador < interacciones:
  years_tienen = int(input())
  years = []
  years_wint_spaces = input()
  years = years_wint_spaces.split(" ")
  year_alls =1+ int(max(years)) - int(min(years))
  salidad_data=salidad_data+str(year_alls-years_tienen)+"\n"
  contador += 1
print(salidad_data)
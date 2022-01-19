import csv
import string
import requests
from sys import exit
from bs4 import BeautifulSoup
from time import sleep
import re
import numpy as np
import pandas as pd
import os



headers={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0", "Pragma":"no-cache", "Sec-Fetch-Dest": "document", "Sec-Fetch-Mode": "navigate", "Sec-Fetch-Site": "none", "Sec-Fetch-User": "?1", "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8", "Accept-Language": "en-US,en;q=0.5", "Accept-Encoding": "br"}
shburl = "https://www.sahibinden.com"
sahibinden = requests.Session()

ilanlistesi = []

for x in [1]: # just 20 ads for replicable code
#for x in range(1,40): # 2x30=600 webpages(each page is a house for sale) to be datamined
    no=20*x
    try:
        h = sahibinden.get(shburl + "/satilik-daire/istanbul-kartal?pagingOffset="+str(no), headers=headers, timeout=5)
    except Exception as e:
        print("couldnt pull! Hata: ", e)
        exit()
    parser = BeautifulSoup(h.text, "html.parser")
    for link in parser.findAll("a"):
        try:
            if link.get("href").find("/ilan/") != -1 and 'img' not in str(link):
                ilanlistesi.append(link.get("href"))
        except:
            continue
    
    if x%5==0 and x>0:
        print(len(ilanlistesi))
    if x%40==0 and x>0:
        bekleme = 390
    else:
        bekleme = 7
    beklenen = 0
    while beklenen < bekleme:
        sleep(1)
        beklenen += 1
        print("[RATELIMIT BYPASS] ", bekleme, " seconds you have to wait. Beklenen:", beklenen)
        
          # Sahibinden.com blocks IP after multiple requests 
          # in a certain time period (40 reqs in 5 min approx)

data = {}
x=0
for sayac,ilan in enumerate(ilanlistesi[1:]):
#for sayac,ilan in enumerate(ilanlistesi[:9]):
    try:
        h = sahibinden.get(shburl + ilan, headers=headers, timeout=10)
    except:
        print("[ERROR]", ilan, " cekilemedi!")
        continue
    text = h.text
    text = text.replace("\n", "")
    text = text.replace("\r", "")
    text = text.replace("\t", "")
    text = text.replace("\n", "")
    text = text.replace("&nbsp;", "")
    text = text.replace("  ", "")
    matches = re.findall(r'>(.*?)<', text)
    
    ilaninfo = {'ucret': '', 'Mahalle': '', 'ilanno': '',\
                'ilantarih': '', 'emlaktipi': '', 'brutmetrekare': '', \
                'netmetrekare': '', 'odasayi': '', 'binayasi': '', 'bulundugukat': '', \
                'katsayisi': '', 'isitma': '', 'dogalgazkombi': '', 'banyosayisi': '', \
                'banyo': '', 'esyali': '', 'kullanimdurumu': '', 'siteicerisinde': '', \
                'siteadi': '', 'aidat': '', 'krediyeuygun': '', 'tapudurumu': '',\
                'kimden': '', 'goruntuluaramailegezilebilir': '', 'takas': ''}
    for index, tags in enumerate(matches):
        ilanToplandi = False
        if len(tags) > 35:
            continue
        
        if tags == "İlan No":
            ilanToplandi = True
            ilaninfo["ilanno"] = matches[index + 2]
        elif tags == "İlan Tarihi":
            ilanToplandi = True
            ilaninfo["ilantarih"] = matches[index + 2]
        elif tags == "Emlak Tipi":
            ilanToplandi = True
            ilaninfo["emlaktipi"] = matches[index + 2]
        elif tags == "m² (Brüt)":
            ilanToplandi = True
            ilaninfo["brutmetrekare"] = matches[index + 2]
        elif tags == "m² (Net)":
            ilanToplandi = True
            ilaninfo["netmetrekare"] = matches[index + 2]
        elif tags == "Oda Sayısı":
            ilanToplandi = True
            ilaninfo["odasayi"] = matches[index + 2]
        elif tags == "Bina Yaşı":
            ilanToplandi = True
            ilaninfo["binayasi"] = matches[index + 2]
        elif tags == "Bulunduğu Kat":
            ilanToplandi = True
            ilaninfo["bulundugukat"] = matches[index + 2]
        elif tags == "Kat Sayısı":
            ilanToplandi = True
            ilaninfo["katsayisi"] = matches[index + 2]
        elif tags == "Isıtma":
            ilanToplandi = True
            ilaninfo["isitma"] = matches[index + 2]
        elif tags == "Doğalgaz (Kombi)":
            ilanToplandi = True
            ilaninfo["dogalgazkombi"] = matches[index + 2]
        elif tags == "Banyo Sayısı":
            ilanToplandi = True
            ilaninfo["banyosayisi"] = matches[index + 2]
        elif tags == "Balkon":
            ilanToplandi = True
            ilaninfo["banyo"] = matches[index + 2]
        elif tags == "Eşyalı":
            ilanToplandi = True
            ilaninfo["esyali"] = matches[index + 2]
        elif tags == "Kullanım Durumu":
            ilanToplandi = True
            ilaninfo["kullanimdurumu"] = matches[index + 2]
        elif tags == "Site İçerisinde":
            ilanToplandi = True
            ilaninfo["siteicerisinde"] = matches[index + 2]
        elif tags == "Site Adı":
            ilanToplandi = True
            ilaninfo["siteadi"] = matches[index + 2]
        elif tags == "Aidat (TL)":
            ilanToplandi = True
            ilaninfo["aidat"] = matches[index + 2]
        elif tags == "Krediye Uygun":
            ilanToplandi = True
            ilaninfo["krediyeuygun"] = matches[index + 2]
        elif tags == "Kimden":
            ilanToplandi = True
            ilaninfo["kimden"] = matches[index + 2]
        elif tags == "Görüntülü Arama İle Gezilebilir":
            ilanToplandi = True
            ilaninfo["goruntuluaramailegezilebilir"] = matches[index + 2]
        elif tags == "Takas":
            ilanToplandi = True
            ilaninfo["takas"] = matches[index + 2]
        elif tags.find(" TL") != -1:
            ilanToplandi = True
            ilaninfo["ucret"] = tags.strip()
        elif tags == "Tapu Durumu":
            ilanToplandi = True
            ilaninfo["tapudurumu"] = matches[index + 2]
        elif tags == "Kat Karşılığı":
            ilanToplandi = True
            ilaninfo["katkarsiligi"] = matches[index + 2]
        elif tags == "Krediye Uygunluk":
            ilanToplandi = True
            ilaninfo["krediyeuygunluk"] = matches[index + 2]
        elif 'Mah.' in tags:
            ilanToplandi = True
            ilaninfo["Mahalle"] = matches[index]
        elif 'Mh.' in tags:
            ilanToplandi = True
            ilaninfo["Mahalle"] = matches[index]
    
    if not ilanToplandi:
        print("[" + ilan + " "+str(sayac)+" rows xollected ! next iteration..."+" "+str(ilaninfo.get('ucret')))
        data[ilan] = ilaninfo
    else:
        print("[" + ilan + "] - VERI TOPLANIRKEN HATA GERCEKLESTI! Sonraki iterasyona geciliyor...")
    #if str(ilaninfo.get('ucret'))=="":
    #       break
    bekleme = 1
    beklenen = 0
    while beklenen < bekleme:
        sleep(2)
        beklenen += 1
        print("[RATELIMIT BYPASS] ", bekleme, " seconds you have to wait. total waited:", beklenen)
    #if sayac%30==0 and sayac>0:
     #   sleep(410)
      #  print("330 SECONDS LONG PAUSE EVERY 30 OBSERVATIONS")
    if sayac%50==0 and sayac>0:
        list_of_list=[]
        for x,z in data.items():
            a=[x]
            #k='/detay'
            #a.append(x[x.rfind('-')+1:x.rfind(k)])
            for g in z.values():
                a.append(g)
            list_of_list.append(a)
        columnss=['ad_link','price', 'neighborhood', 'ad_id', 'ad_date', 'real_estate_type', 'square_meter_total', 'square_meter_net', 'number_of_rooms', 'age_building', 'which_floor', 'total_numberof_floors', 'heating_system', 'naturalgas_avb', 'bathrooms', 'bathroom_yes_no', 'house_appliances', 'usage_st', 'in_a_compound', 'compound_name', 'monthly_payment', 'loan_status', 'realestate_document_type', 'seller_type', 'online_tour_available', 'trade']
        df = pd.DataFrame(list_of_list, columns=columnss)
        df.to_csv('SAMPLE_REALESTATE.csv')


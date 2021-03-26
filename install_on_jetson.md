# Installing everything needed for jetson operation

Na początku klonujemy to repo, ja zrobiłem w $HOME i taką lokalizacje zakładam dalej.
```
git clone https://github.com/jkjung-avt/tensorrt_demos
```
Następnie chcemy dodać cude do PATH bo JetPack nie ma tych zmiennych domyślnie dodanych
(unikamy błędów typu: nvcc not found etc.)
```
echo "export PATH=/usr/local/cuda/bin\${PATH:+:\${PATH}}" >> ${HOME}/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}" >> ${HOME}/.bashrc
```

Jetpack też nie ma pip3 więc jazda:
```
sudo apt-get install python3-pip
```

Powyższe repo wymaga protobuf, od razu też instalujemy onnx
```
pip3 install protobuf
sudo pip3 install onnx==1.4.1
```

Przechodzimy do katalogu plugins i robimy make
```
cd $HOME/tensorrt_demos/plugins
make 
```

Teraz instalujemy pycuda, skryptem z katalogu ssd
```
cd $HOME/tensorrt_demos/ssd
./install_pycuda.sh
```

## Używanie repo do konwersji modeli

W katalogu $HOME/tensorrt_demos/yolo:
```
python3 yolo_to_onnx.py -m <nazwa_modelu_bez_rozszerzen> -c 1 
python3 onnx_to_tensorrt.py -m <nazwa_modelu_bez_rozszerzen> -c 1
```
Dodajemy argument -c 1, żeby nie krzyczał że nia mamy 80 klas w modelu (COCO tyle ma), w przypadku wykrywania pachołków mamy tylko 1. 
Przykładowo: 
```
python3 yolo_to_onnx.py -m yolov4-416 -c 1 
python3 onnx_to_tensorrt.py -m yolov4-416 -c 1
```
Modele muszą się znajdować w tym samym katalogu co skrypty. Powyższym sposobem przekształcimy yolov4 do trt FP16.

## Ewaluacja
Umieszczamy pliki z datasetu w folderze cocoapi/images/dataset w $HOME. Ważna jest żeby istniał pośredni folder "dataset", ponieważ plik walidacyjny odnośni się do nich przez "dataset/img_xxxxx". 
Ewaluacje robimy z katalogu tensorrt_demos/, modele powinny znajdować się w katalogu yolo.

Przyda się pycocotools, progresbar2 
```
pip3 install pycocotools
pip3 install progressbar2
```

Wykonujemy delikatną modyfikacji skryptu yolo_eval.py: https://pastebin.pl/view/084b1a90 (wstaw to tutaj)
Ważne jest żeby zmienić parametry areaRNG aby odwzorowywały dobre rozmiary. 
Też trzeba się upewnić czy pole area w val.json nie jest 0. :) 


<!-- Pycocotools może nie chcieć się instalować z pipa wtedy: -->
<!-- ``` -->
<!-- pip3 install cython -->
<!-- git clone https://github.com/cocodataset/cocoapi -->
<!-- cd $HOME/cocoapi/PythonAPI -->
<!-- sudo make install  -->
<!-- ``` -->

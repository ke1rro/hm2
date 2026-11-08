# Результати детекторів і контекст для обговорення data augmentation

Дата зрізу: 2026-11-07. Це аналітична довідка для планування наступних експериментів і передачі контексту ChatGPT, а не фінальний студентський звіт для LMS. Нове навчання та augmentation під час її підготовки не запускалися.

## 1. Джерела та як читати числа

- Validation: оригінальні `config.json`, `status.json`, `metrics.json`, per-epoch records та environment, отримані з Google Drive. Faster R-CNN generation `1794008813488173932`, summary `1794008838839340466`; SHA256 усіх файлів і checkpoint перевірено. RetinaNet: завершений batch `retinanet_stride5_full_5epochs_seed2026`, локальні оригінальні записи з provenance.
- [RetinaNet: архів результатів на Drive](https://drive.google.com/file/d/1e5AybR4U8T-SMt2nJ8XcipM_2ye5COUn/view). [Faster R-CNN: checkpoint і записи](https://drive.google.com/drive/folders/10pqkOV1t-XlHyv4S2KR7IJuqUg9eoiAy). [Faster R-CNN: summary з CSV](https://drive.google.com/drive/folders/11-kfw9cFvQkfI4KdmAfDTGWmLy0Ha-hj).
- Public scores переписані з наданого користувачем скриншота Kaggle submissions від 2026-11-07. Їх не отримано незалежним запитом до Kaggle API. Private scores ще невідомі.
- EDA заново агреговано з `data/metadata/frames.parquet`, `objects.parquet`, `ignored_regions.parquet`. Бокси рахуються як анотації по кадрах, не як унікальні об'єкти: той самий об'єкт повторюється у відео.
- AP подано в шкалі 0–1. Різниця 0.01 AP дорівнює одному процентному пункту (п.п.). Public score не є accuracy. Точна формула custom-метрики Kaggle й геометрична семантика ignore не підтверджені, тому Public score та validation COCO mAP не ототожнюємо.
- Усі наведені якісні відмінності є описовими результатами одного seed. Статистичну значущість не перевіряли.

## 2. Дані, дисбаланс і масштаб

VisDrone2019-VID, 8 класів. Офіційний validation повністю відділений від train; випадковий split по кадрах неприпустимий через сильну часову кореляцію. Test не має доступних GT. Public/private поділ у змаганні виконується за кліпами.

| Split | Кадри | Кліпи | Розміри, min → max   | GT boxes |
| ----- | ----- | ----- | -------------------- | -------- |
| train | 24201 | 56    | 1344×756 → 2720×1530 | 1105516  |
| val   | 2846  | 7     | 1344×756 → 3840×2160 | 114132   |
| test  | 3811  | 48    | 680×382 → 1920×1080  | невідомі |

Розміри вище виміряні з метаданих, а не взяті лише із загального опису змагання. Train і validation містять кадри більші за test. Самі test-метадані не використовуються для вибору augmentation або моделі.

| Split | Об’єктів/кадр: mean / median | P90 / P99 / max | Порожні labeled кадри | Кадри з ignore |
| ----- | ---------------------------- | --------------- | --------------------- | -------------- |
| train | 45.68 / 43                   | 92 / 142 / 180  | 3                     | 18090 (74.75%) |
| val   | 40.10 / 37                   | 81 / 113 / 116  | 0                     | 1434 (50.39%)  |

Щільність сильно відрізняється між train-кліпами: середнє від 5.15 до 142.96 об’єкта/кадр. Тому випадкові кадри не є незалежними спостереженнями. Не слід інтерпретувати тисячі GT одного класу як тисячі незалежних сцен.

| Клас     | Train boxes | Train % | Stride-5 boxes | Val boxes | Val % | Train median w×h, px |
| -------- | ----------- | ------- | -------------- | --------- | ----- | -------------------- |
| person   | 328701      | 29.73   | 66353          | 50312     | 44.08 | 25×50                |
| bicycle  | 40255       | 3.64    | 8129           | 6022      | 5.28  | 43×43                |
| car      | 505301      | 45.71   | 101814         | 31821     | 27.88 | 67×48                |
| van      | 46940       | 4.25    | 9452           | 6842      | 5.99  | 55×46                |
| truck    | 30498       | 2.76    | 6137           | 1359      | 1.19  | 79×68                |
| tricycle | 41349       | 3.74    | 8337           | 5487      | 4.81  | 59×51                |
| bus      | 9653        | 0.87    | 1945           | 264       | 0.23  | 121×103              |
| motor    | 102819      | 9.30    | 20712          | 12025     | 10.54 | 38×43                |

Car і person домінують у train; bus є найрідкіснішим. Частоти класів і типовий масштаб змінюються між train та validation. Стрибок AP рідкісного класу може бути нестабільним через мале число validation-сцен; до повторного семплінгу треба переглянути приклади та покриття кліпів.

| Split | Small, площа \<32² | Medium, 32²–\<96² | Large, ≥96² | Median width / height / area |
| ----- | ------------------ | ----------------- | ----------- | ---------------------------- |
| train | 24.91%             | 63.50%            | 11.59%      | 44 / 48 / 2100               |
| val   | 19.20%             | 69.53%            | 11.27%      | 47 / 57 / 2745               |

Ці size bins визначені в оригінальних пікселях, як COCO AP_small. Додатково пораховано приблизний масштаб після resize 640/1024:

$$
s = \\min\\left(\\frac{640}{\\min(H,W)},\\frac{1024}{\\max(H,W)}\\right).
$$

Обчислення не включає цілочисельне округлення resize і padding. Воно описує розміри вхідних об’єктів, а не перевизначає COCO категорії площі.

| Split | Площа після resize \<32² | Коротша сторона \<8 px | Коротша сторона \<4 px |
| ----- | ------------------------ | ---------------------- | ---------------------- |
| train | 69.16%                   | 4.33%                  | 0.44%                  |
| val   | 66.11%                   | 5.33%                  | 0.37%                  |

Це аргумент перевіряти втрату дрібних деталей при зменшенні масштабу. Збільшення картинки після втрати деталей не повертає їх. Однак сам цей підрахунок не доводить, що crop, вища роздільність або інші anchors покращать AP.

Ignored regions: 63 843 унікальні геометрії в train і 3 995 у validation. У COCO вони повторені по категоріях: 510 744 і 31 960 crowd-records відповідно; це не окремі області. Для training це don't-care, а не додатковий background. Валідні GT всередині ignored області не можна просто видаляти. Частка площі union ignored regions не виміряна.

| Occlusion raw code | Train boxes | Val boxes |
| ------------------ | ----------- | --------- |
| 0                  | 600789      | 46728     |
| 1                  | 445799      | 59822     |
| 2                  | 58928       | 7582      |

Occlusion/truncation збережені як raw metadata codes. Вони ще не використані для augmentation-aware sampling; значення кодів не інтерпретуємо як ступені без перевірки source encoding. Integrity-перевірки метаданих знайшли відсутній `sample_submission.csv`; інші перевірки, включно з координатами та наявністю зображень, пройшли.

## 3. Фактичний протокол і augmentation baseline

Спільне: COCO-pretrained ResNet-50 FPN V2; seed 2026; batch size 2; 5 епох; SGD, lr=0.0025, momentum=0.9, weight_decay=0.0001; AMP на CUDA; resize зі збереженням aspect ratio, short side 640 і max long side 1024; стандартна нормалізація TorchVision з pretrained моделі. Усі записані середовища мають Tesla T4, Python 3.13.16, torch 2.11.0+cu130, torchvision 0.26.0+cu130. RetinaNet працював через Colab/Kaggle й міг продовжуватися між сесіями; це не контрольований benchmark апаратного часу.

Єдина реалізована випадкова image augmentation в цих запусках: **horizontal flip з p=0.5**, лише на train. Немає color jitter, blur, crops, mosaic, mixup, copy-paste або випадкового multiscale training. Бокси й ignored regions у Faster R-CNN перевертаються разом із зображенням. Validation/test мають deterministic preprocessing.

LR scheduler знижує LR у точках приблизно 2/3 та 90% запланованих updates. Warmup: 100 updates для stride-5 і Faster R-CNN; для full RetinaNet близько 497 updates, оскільки warmup збережено в частках епохи. Це ще одна відмінність бюджетів full і subset.

RetinaNet: Focal classification + GIoU regression. Standard P3 sizes [32,40,50], smaller [16,20,25]; решта рівнів також зменшена приблизно вдвічі, ratios [0.5,1,2] незмінні. Inference score threshold=0.05, NMS=0.5, cap=300. Stock training не виключає негативні anchors в ignored regions.

Faster R-CNN: RPN binary cross-entropy + Smooth L1, RoI cross-entropy + Smooth L1. Ignore-aware sampling виключає негативні anchors/proposals із coverage ≥0.5 (частка площі самого anchor/proposal, перекрита однією ignored region), залишає positives. RPN proposals: pre/post NMS train 2000, test 1000; RoI sample size 512, positive fraction 0.25; sizes [32,64,128,256,512]; detections cap=500; trainable backbone layers=3.

Exact ordered train manifest Faster R-CNN збігається зі stride-5 manifest RetinaNet; хеші `dataset_selection_sha256` між runner-ами мають різні алгоритми побудови й не повинні порівнюватися як однаковий контракт.

Спільна validation: всі 2 846 кадрів, original-image boxes, COCO IoU 0.50:0.05:0.95, 8 класів, original category-specific iscrowd. Основне порівняння maxDets=[1,10,100]. Faster R-CNN додатково оцінений при maxDets=[1,10,500]. Різниця detector family одночасно змінила ignore training і частину head/inference settings, тому це не чиста абляція proposal stage.

## 4. Validation усіх фінальних моделей, epoch 5

| Модель                        | mAP      | AP50     | AP75     | AP small | AP medium | AP large |
| ----------------------------- | -------- | -------- | -------- | -------- | --------- | -------- |
| RetinaNet standard / stride 5 | 0.152786 | 0.291178 | 0.141840 | 0.053135 | 0.131627  | 0.316873 |
| RetinaNet smaller / stride 5  | 0.148782 | 0.316268 | 0.123687 | 0.052404 | 0.130756  | 0.276468 |
| RetinaNet standard / full     | 0.150543 | 0.285836 | 0.138214 | 0.049999 | 0.127531  | 0.320663 |
| RetinaNet smaller / full      | 0.156314 | 0.324951 | 0.134056 | 0.054759 | 0.142678  | 0.303211 |
| Faster R-CNN / stride 5       | 0.174703 | 0.353521 | 0.152856 | 0.080675 | 0.168725  | 0.334301 |

Faster R-CNN при maxDets=500: map=0.174703287, ap50=0.353521097, ap75=0.152856441, ap_small=0.080674981, ap_medium=0.168725513, ap_large=0.334300749. Майже тотожний агрегований score при 100 і 500 у цьому запуску не доводить, що ліміт не важливий в інших моделях/сценах.

## 5. Реальні submissions і відповідні checkpoint-и

RetinaNet CSV використовують best validation checkpoint, Faster R-CNN використовує налаштовану фінальну epoch 5. Саме тому для full RetinaNet нижче інші validation-числа, ніж у таблиці epoch 5. Вибір checkpoint не робився за Public score.

| Модель                        | Submitted epoch | Validation mAP | AP50     | AP75     | AP small | AP medium | AP large | Kaggle Public |
| ----------------------------- | --------------- | -------------- | -------- | -------- | -------- | --------- | -------- | ------------- |
| RetinaNet standard / stride 5 | 5               | 0.152786       | 0.291178 | 0.141840 | 0.053135 | 0.131627  | 0.316873 | 0.16010       |
| RetinaNet smaller / stride 5  | 5               | 0.148782       | 0.316268 | 0.123687 | 0.052404 | 0.130756  | 0.276468 | 0.14202       |
| RetinaNet standard / full     | 1               | 0.157504       | 0.299334 | 0.148558 | 0.052500 | 0.131319  | 0.327028 | 0.16826       |
| RetinaNet smaller / full      | 4               | 0.158831       | 0.333095 | 0.133870 | 0.058264 | 0.145968  | 0.306402 | 0.15983       |
| Faster R-CNN / stride 5       | 5               | 0.174703       | 0.353521 | 0.152856 | 0.080675 | 0.168725  | 0.334301 | 0.20369       |

| CSV                              | Рядки | Detections | Max/image | Порожні |
| -------------------------------- | ----- | ---------- | --------- | ------- |
| standard_anchors_stride5.csv     | 3811  | 522251     | 265       | 0       |
| smaller_anchors_stride5.csv      | 3811  | 1061204    | 300       | 0       |
| standard_anchors_full.csv        | 3811  | 520292     | 261       | 0       |
| smaller_anchors_full.csv         | 3811  | 876729     | 300       | 0       |
| faster_rcnn_baseline_stride5.csv | 3811  | 287745     | 142       | 0       |

Усі CSV мають класи 0–7, original pixel top-left xywh, один рядок на test image. Для Faster R-CNN повторно перевірено точний порядок `test_images.csv`, межі боксу та SHA256 з Drive summary; повторний inference не проводився. Офіційний sample відсутній, `official_sample_verified=false`. Старий `submission_unverified.csv` зі score 0.16010 на скриншоті є попереднім експортом standard-anchor stride-5 RetinaNet, а не шостим незалежним експериментом.

## 6. Час, бюджет і швидкість

Training time нижче є сумою end-to-end latency training batches із status, включно з batch decode/transfer/forward/backward. Не включає setup, validation, checkpoint upload, pause між сесіями або test export. Не називаємо його повним wall-clock часом експерименту.

| Модель                        | Train frames | Exposures | Updates | AMP skips | Training min | Mean train min/epoch | Validation sum min | Train images/s |
| ----------------------------- | ------------ | --------- | ------- | --------- | ------------ | -------------------- | ------------------ | -------------- |
| RetinaNet standard / stride 5 | 4868         | 24340     | 12165   | 5         | 59.61        | 11.92                | 40.96              | 6.81           |
| RetinaNet smaller / stride 5  | 4868         | 24340     | 12166   | 4         | 60.04        | 12.01                | 45.51              | 6.76           |
| RetinaNet standard / full     | 24201        | 121005    | 60484   | 21        | 284.18       | 56.84                | 29.72              | 7.10           |
| RetinaNet smaller / full      | 24201        | 121005    | 60485   | 20        | 283.14       | 56.63                | 28.80              | 7.12           |
| Faster R-CNN / stride 5       | 4868         | 24340     | 12166   | 4         | 75.63        | 15.13                | 34.74              | 5.36           |

За повним наданим Kaggle log Faster R-CNN від початку notebook до завершення конвертації минуло приблизно 7 593 s, або **126.5 хв**. Із них training batches 75.63 хв та validation 34.74 хв; решта припадає на setup, checkpoint transfers, speed probe, test inference/export і conversion. `chunk_seconds` у status є часом окремого runner invocation, а не універсальним total timer. Надійний повний active wall-clock для кожного RetinaNet через зміни сесій не відновлено.

| Модель epoch 5                | Val batch | Val images/s | Val ms/frame | Frames | GPU      |
| ----------------------------- | --------- | ------------ | ------------ | ------ | -------- |
| RetinaNet standard / stride 5 | 2         | 8.04         | 124.38       | 2846   | Tesla T4 |
| RetinaNet smaller / stride 5  | 2         | 8.17         | 122.45       | 2846   | Tesla T4 |
| RetinaNet standard / full     | 2         | 15.04        | 66.49        | 2846   | Tesla T4 |
| RetinaNet smaller / full      | 2         | 14.39        | 69.51        | 2846   | Tesla T4 |
| Faster R-CNN / stride 5       | 2         | 11.20        | 89.32        | 2846   | Tesla T4 |

Faster R-CNN додатковий inference benchmark: batch 1 на 300 val frames, 10.62 images/s, 94.17 ms/frame. Test export: batch 2, 16.76 images/s, 59.67 ms/frame. У всіх швидкостях враховані decode, collation, transfer, model transform, inference, NMS і CPU outputs; COCO evaluation виключена; один warmup batch. Час різних сесій не можна трактувати як гарантовану перевагу архітектури. Повної зіставної VRAM-таблиці для всіх п'яти завершених запусків немає.

## 7. Per-class AP, canonical maxDets=100

Фінальні моделі epoch 5:

| Клас     | RetinaNet standard / stride 5 | RetinaNet smaller / stride 5 | RetinaNet standard / full | RetinaNet smaller / full | Faster R-CNN / stride 5 |
| -------- | ----------------------------- | ---------------------------- | ------------------------- | ------------------------ | ----------------------- |
| person   | 0.203834                      | 0.211988                     | 0.205685                  | 0.215429                 | 0.247199                |
| bicycle  | 0.106394                      | 0.133531                     | 0.076764                  | 0.128947                 | 0.148030                |
| car      | 0.339475                      | 0.328795                     | 0.331044                  | 0.333632                 | 0.356788                |
| van      | 0.164058                      | 0.152070                     | 0.154683                  | 0.170766                 | 0.236865                |
| truck    | 0.162982                      | 0.155979                     | 0.147277                  | 0.164201                 | 0.139097                |
| tricycle | 0.140784                      | 0.120347                     | 0.128633                  | 0.134819                 | 0.164380                |
| bus      | 0.037640                      | 0.008979                     | 0.086707                  | 0.013251                 | 0.014804                |
| motor    | 0.067124                      | 0.078563                     | 0.073550                  | 0.089463                 | 0.090464                |

Checkpoint-и, які реально пішли в submissions:

| Клас     | RetinaNet standard / stride 5 | RetinaNet smaller / stride 5 | RetinaNet standard / full | RetinaNet smaller / full | Faster R-CNN / stride 5 |
| -------- | ----------------------------- | ---------------------------- | ------------------------- | ------------------------ | ----------------------- |
| person   | 0.203834                      | 0.211988                     | 0.207135                  | 0.213616                 | 0.247199                |
| bicycle  | 0.106394                      | 0.133531                     | 0.097018                  | 0.132770                 | 0.148030                |
| car      | 0.339475                      | 0.328795                     | 0.342950                  | 0.337827                 | 0.356788                |
| van      | 0.164058                      | 0.152070                     | 0.173080                  | 0.176078                 | 0.236865                |
| truck    | 0.162982                      | 0.155979                     | 0.178555                  | 0.161937                 | 0.139097                |
| tricycle | 0.140784                      | 0.120347                     | 0.147514                  | 0.137628                 | 0.164380                |
| bus      | 0.037640                      | 0.008979                     | 0.044106                  | 0.017440                 | 0.014804                |
| motor    | 0.067124                      | 0.078563                     | 0.069673                  | 0.093354                 | 0.090464                |

## 8. Усі per-epoch validation metrics, maxDets=100

| Модель                        | Epoch | mAP      | AP50     | AP75     | AP small | AP medium | AP large | Training cumulative min | Validation s |
| ----------------------------- | ----- | -------- | -------- | -------- | -------- | --------- | -------- | ----------------------- | ------------ |
| RetinaNet standard / stride 5 | 1     | 0.136029 | 0.260485 | 0.127122 | 0.042739 | 0.116523  | 0.285436 | 11.70                   | 500.8        |
| RetinaNet standard / stride 5 | 2     | 0.139080 | 0.267904 | 0.130385 | 0.039200 | 0.129469  | 0.279288 | 23.37                   | 451.8        |
| RetinaNet standard / stride 5 | 3     | 0.141832 | 0.276390 | 0.129733 | 0.045163 | 0.126128  | 0.297394 | 35.40                   | 517.0        |
| RetinaNet standard / stride 5 | 4     | 0.145679 | 0.281275 | 0.133211 | 0.052310 | 0.126253  | 0.306901 | 47.46                   | 491.4        |
| RetinaNet standard / stride 5 | 5     | 0.152786 | 0.291178 | 0.141840 | 0.053135 | 0.131627  | 0.316873 | 59.61                   | 496.3        |
| RetinaNet smaller / stride 5  | 1     | 0.119191 | 0.258900 | 0.094754 | 0.047807 | 0.102474  | 0.219467 | 11.81                   | 556.9        |
| RetinaNet smaller / stride 5  | 2     | 0.133283 | 0.288994 | 0.107391 | 0.046866 | 0.119175  | 0.242236 | 23.87                   | 536.7        |
| RetinaNet smaller / stride 5  | 3     | 0.141500 | 0.305928 | 0.112798 | 0.052007 | 0.125878  | 0.262876 | 35.98                   | 556.0        |
| RetinaNet smaller / stride 5  | 4     | 0.145141 | 0.311666 | 0.118697 | 0.054249 | 0.126802  | 0.274691 | 47.94                   | 545.4        |
| RetinaNet smaller / stride 5  | 5     | 0.148782 | 0.316268 | 0.123687 | 0.052404 | 0.130756  | 0.276468 | 60.04                   | 535.3        |
| RetinaNet standard / full     | 1     | 0.157504 | 0.299334 | 0.148558 | 0.052500 | 0.131319  | 0.327028 | 60.47                   | 499.6        |
| RetinaNet standard / full     | 2     | 0.147781 | 0.285560 | 0.136185 | 0.049026 | 0.132283  | 0.310660 | 115.98                  | 311.9        |
| RetinaNet standard / full     | 3     | 0.150630 | 0.290765 | 0.137615 | 0.052711 | 0.128217  | 0.320635 | 171.67                  | 367.0        |
| RetinaNet standard / full     | 4     | 0.152203 | 0.290653 | 0.138295 | 0.049786 | 0.130558  | 0.326204 | 227.93                  | 311.3        |
| RetinaNet standard / full     | 5     | 0.150543 | 0.285836 | 0.138214 | 0.049999 | 0.127531  | 0.320663 | 284.18                  | 293.2        |
| RetinaNet smaller / full      | 1     | 0.155872 | 0.334016 | 0.128373 | 0.054925 | 0.138426  | 0.289509 | 56.50                   | 353.5        |
| RetinaNet smaller / full      | 2     | 0.157198 | 0.332833 | 0.129729 | 0.060823 | 0.144587  | 0.293741 | 112.93                  | 348.6        |
| RetinaNet smaller / full      | 3     | 0.152046 | 0.319589 | 0.127932 | 0.056415 | 0.138447  | 0.295240 | 169.38                  | 348.2        |
| RetinaNet smaller / full      | 4     | 0.158831 | 0.333095 | 0.133870 | 0.058264 | 0.145968  | 0.306402 | 225.70                  | 336.3        |
| RetinaNet smaller / full      | 5     | 0.156314 | 0.324951 | 0.134056 | 0.054759 | 0.142678  | 0.303211 | 283.14                  | 341.6        |
| Faster R-CNN / stride 5       | 1     | 0.192021 | 0.398392 | 0.161919 | 0.089808 | 0.177219  | 0.354186 | 15.26                   | 432.2        |
| Faster R-CNN / stride 5       | 2     | 0.185873 | 0.377485 | 0.161392 | 0.078664 | 0.172655  | 0.358500 | 30.39                   | 431.2        |
| Faster R-CNN / stride 5       | 3     | 0.178387 | 0.358400 | 0.155652 | 0.090534 | 0.167787  | 0.327530 | 45.48                   | 409.9        |
| Faster R-CNN / stride 5       | 4     | 0.178642 | 0.358396 | 0.158583 | 0.084534 | 0.167881  | 0.342349 | 60.55                   | 414.1        |
| Faster R-CNN / stride 5       | 5     | 0.174703 | 0.353521 | 0.152856 | 0.080675 | 0.168725  | 0.334301 | 75.63                   | 397.3        |

### Training losses: середні за epoch 1 і 5

RetinaNet: weighted mean компактних вікон, де вага дорівнює `batch_count`; Faster R-CNN: mean оригінальних batch records. Це середні за batches, не за кількістю GT. Змішані компоненти різних моделей не порівнюємо чисельно між собою.

| Модель                        | Epoch | Total    | Classification | Box regression | RPN objectness | RPN regression |
| ----------------------------- | ----- | -------- | -------------- | -------------- | -------------- | -------------- |
| RetinaNet standard / stride 5 | 1     | 0.618128 | 0.339644       | 0.278484       | —              | —              |
| RetinaNet standard / stride 5 | 5     | 0.408706 | 0.167416       | 0.241291       | —              | —              |
| RetinaNet smaller / stride 5  | 1     | 0.715191 | 0.403196       | 0.311995       | —              | —              |
| RetinaNet smaller / stride 5  | 5     | 0.465410 | 0.217742       | 0.247668       | —              | —              |
| RetinaNet standard / full     | 1     | 0.492352 | 0.236211       | 0.256141       | —              | —              |
| RetinaNet standard / full     | 5     | 0.314338 | 0.111390       | 0.202948       | —              | —              |
| RetinaNet smaller / full      | 1     | 0.557298 | 0.286253       | 0.271045       | —              | —              |
| RetinaNet smaller / full      | 5     | 0.334735 | 0.129131       | 0.205604       | —              | —              |
| Faster R-CNN / stride 5       | 1     | 0.893081 | 0.290305       | 0.382433       | 0.092483       | 0.127861       |
| Faster R-CNN / stride 5       | 5     | 0.534976 | 0.146909       | 0.276143       | 0.032250       | 0.079675       |

## 9. Що результати дозволяють стверджувати

| Порівняння epoch 5, п.п.                    | mAP     | AP50    | AP75    | AP small | AP medium | AP large |
| ------------------------------------------- | ------- | ------- | ------- | -------- | --------- | -------- |
| Smaller − standard, stride 5                | -0.4005 | +2.5090 | -1.8153 | -0.0730  | -0.0871   | -4.0406  |
| Smaller − standard, full                    | +0.5771 | +3.9115 | -0.4158 | +0.4760  | +1.5146   | -1.7452  |
| Full − stride 5, standard                   | -0.2243 | -0.5342 | -0.3627 | -0.3136  | -0.4095   | +0.3790  |
| Full − stride 5, smaller                    | +0.7532 | +0.8682 | +1.0369 | +0.2354  | +1.1922   | +2.6743  |
| Faster R-CNN − RetinaNet standard, stride 5 | +2.1917 | +6.2343 | +1.1016 | +2.7540  | +3.7099   | +1.7427  |

- Менші anchors не дали універсального покращення: на stride-5 mAP і AP_small трохи нижчі, а AP50 вища. Розрив AP50/AP75 узгоджується з гіпотезою обмеженої точної локалізації, але не встановлює її причину.
- Для standard anchors full-data модель має найкращу val epoch 1; для smaller anchors full-data найкраща epoch 4. Погіршення до epoch 5 не є саме по собі доказом overfitting: також різняться update budget і scheduler.
- На Public leaderboard повні дані покращили обидва RetinaNet варіанти: +0.00816 score для standard та +0.01781 для smaller. На validation epoch 5 standard full гірший, smaller full кращий. Отже, твердження «повні дані зовсім не допомогли» неправильне.
- Користувач вирішив використовувати кожен п’ятий кадр надалі як компроміс між compute і якістю. Full-data режим має приблизно п’ять разів більше exposures/updates, а не контрольовану різницю лише data diversity. Обидва режими зберігають усі train-кліпи.
- Faster R-CNN має вищі агреговані validation метрики та Public score, але не кращий AP кожного класу. Зокрема bus і truck не мають універсальної переваги. Різницю не можна причинно приписати лише proposal stage, адже змінилася обробка ignored negatives та інші detector defaults.
- Loss values між RetinaNet та Faster R-CNN не порівнюємо як спільну шкалу якості: у них різні компоненти. Зменшення training loss не заміняє покращення validation AP.

## 10. Гіпотези для augmentation, ще не результати

Нижче наведено напрямки для обговорення, не виконаний пошук параметрів і не обіцянку покращення.

| Напрям                                                       | Підстава в даних                                                                                    | Що перевірити / ризик                                                                                                                                             |
| ------------------------------------------------------------ | --------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Помірні brightness/contrast/color зміни                      | Різні кліпи та сцени; baseline містить лише flip                                                    | Зберегти видимість маленьких об’єктів; сильне затемнення може погіршити сигнал. Частоту й силу ще не обрано                                                       |
| Обережний scale jitter або crop із подальшим resize          | Близько 69% train boxes після поточного resize менші за 32²                                         | Downscale/mosaic можуть ще зменшити об’єкти; crop може видалити crowded context і annotations; порівняти object retention і AP_small                              |
| Легка деградація зображення, наприклад blur/compression      | Дрібні об’єкти, відеопоходження                                                                     | Реальну частоту blur/compression ще не виміряно. Спочатку переглянути кадри; сильний blur може стерти об’єкт                                                      |
| Class-aware sampling або copy-paste                          | Bus/truck рідкісні, мало val bus                                                                    | Це окрема зміна data distribution, не лише image transform. Контролювати покриття кліпів, фізичний масштаб, перспективу й ignore semantics; не копіювати val/test |
| Перевірка ignore-aware training перед augmentation RetinaNet | Значна частка train кадрів має ignore; baseline RetinaNet вчиться на них як на background negatives | Окрема контрольована абляція, не змішувати виправлення sampling із новою augmentation                                                                             |

Для першої абляції доцільно обговорити лише один помірний photometric фактор або один scale/crop фактор. До навчання переглянути overlay перетворених GT та ignore boxes на sparse, dense, tiny-object та rare-class train прикладах. Порахувати, скільки boxes видалено, скільки класів зникає, і як змінюються розмір та щільність об’єктів. Політика для об’єкта, чий box видалено, але пікселі частково залишились, має бути явною, щоб не створити false background supervision.

Перетворювати image, GT boxes та ignore regions узгоджено; за потреби оновлювати area, canvas та metadata. Не видаляти валідні GT просто через перетин з ignore. Label IDs і pixel/normalized координати мають залишатися коректними. Не застосовувати випадкові augmentations на validation/test.

Зберегти незмінний baseline, seed, stride-5 manifest, validation, optimizer/scheduler, кількість epochs/updates та evaluation cap; міняти один фактор. Якщо transform змінює training time або кількість об’єктів, записати це. Для variable-size/multiscale експерименту окремо вказати зміни compute budget. Не вибирати augmentation за Public score. Для top candidate бажані додаткові seed або оцінка стабільності за незалежними validation-кліпами; поки таких результатів немає.

Офіційні довідкові матеріали для реалізації: [TorchVision detection transforms](https://docs.pytorch.org/vision/0.20/auto_examples/transforms/plot_transforms_e2e.html) описують спільні перетворення image/boxes і санітизацію після crops; [RandomPhotometricDistort](https://docs.pytorch.org/vision/master/generated/torchvision.transforms.v2.RandomPhotometricDistort.html) є прикладом photometric augmentation, не рекомендацією його defaults для цього набору; [Albumentations bbox augmentation](https://albumentations.ai/docs/3-basic-usage/bounding-boxes-augmentations/) пояснює `min_area`/`min_visibility` та ризик втрати всіх boxes після crop. Перевіряти API проти фактичної версії hosted runtime; не встановлювати нову бібліотеку лише заради цього документа.

## 11. Контекст і запит для наступного обговорення

Можна передати цей документ ChatGPT із таким запитом:

> Допоможи обрати одну наступну контрольовану data-augmentation абляцію для drone object detection. Базуйся лише на наведених вимірюваннях. Розділи спостереження, гіпотезу та те, чого ми ще не знаємо. Запропонуй мінімальний діапазон параметрів, правила трансформації GT/ignore regions, візуальні перевірки перед навчанням, критерій прийняття за validation і ризики для tiny/rare objects. Не змінюй одночасно модель, loss, sampler і augmentation. Не вибирай налаштування за public leaderboard, не вигадуй failure cases чи метрики. Врахуй короткий GPU budget і рішення використовувати кожен п’ятий кадр.

Поки не виконано: augmentation ablations, trained DIoU experiment, контрольована ignore-aware RetinaNet абляція, multi-seed significance analysis, TIDE/error decomposition, оцінка blur/освітлення по кліпах. Цей документ не замінює необхідні tuning/loss experiments або студентський фінальний звіт.

## 12. Локальні файли для відтворення фактів

- `notebooks/01_eda.ipynb`: EDA; метадані в `data/metadata/`.
- `notebooks/03_retinanet_results.ipynb`, `results/retinanet/retinanet_stride5_full_5epochs_seed2026/`: RetinaNet measurements і provenance.
- `notebooks/04_faster_rcnn_results.ipynb`, `runs/faster_rcnn/faster_rcnn_stride5_5epochs_seed2026/baseline_stride5/`: Faster R-CNN records і checkpoint.
- `submissions/`: п’ять CSV із наведеними вище іменами.
- `runs/augmentation_handoff/dataset_facts.json`, `experiment_facts.json`: машинозчитуваний зріз для цієї довідки. Дані й weights gitignored; сам документ містить усі основні числа для передачі без цих файлів.

## 13. Завершений YOLO11s і наступна черга Faster R-CNN

YOLO11s завершив п'ять епох на 4,868 stride-five training кадрах із повною validation
на 2,846 кадрах. Канонічний COCO maxDets 100: mAP **0.154990**, AP50 **0.297651**,
AP small **0.092178**. Найкраща validation епоха збіглася з фінальною, epoch 5.
Training time без validation і передач становив **23.71 хв**, весь Kaggle запуск
разом із setup, validation та Drive становив **61.90 хв**.

Користувач засабмітив `yolo_baseline_stride5.csv`; на скриншоті Kaggle від 2026-11-07
Public score становить **0.14348**. Faster R-CNN epoch 5 має validation mAP 0.174703
і Public 0.20369. Отже, поточний YOLO запуск програв цій контрольній моделі.
Це результат конкретного протоколу, а не висновок про всю архітектуру YOLO.

| Епоха | Validation mAP |
| ----- | -------------: |
| 1     |       0.087870 |
| 2     |       0.116074 |
| 3     |       0.123581 |
| 4     |       0.151891 |
| 5     |       0.154990 |

mAP ще зростав до останньої епохи. Недостатній budget або невідповідний для YOLO
learning-rate schedule залишаються гіпотезами. Ми використали native YOLO11s і
CIoU/BCE/DFL у спільному SGD/AMP loop, а не native Ultralytics Trainer: його стандартний
augmentation stack, EMA та optimizer recipe не застосовувалися. Це можливі фактори,
але причинний висновок без окремої абляції був би необґрунтованим. Порівняння native
validation з поточним adapter на тих самих вагах ще не виконували.

SHA256 checkpoint звірено з Drive manifest; CSV містить усі 3,811 test-зображень,
класи 0–7, коректні source-pixel координати, максимум 216 detections на зображення.
Ці перевірки не виявили помилки формату чи class mapping, але не доводять правильність
усього training protocol. Формула competition scoring залишається відмінною або
непідтвердженою щодо нашого COCO evaluator. Не пояснюємо розрив лише score-ом Public.

[Оригінальні записи YOLO](https://drive.google.com/drive/folders/17zwMD_8UMmdhv1UhHpacDD2SZt6HKAWU),
[CSV](https://drive.google.com/file/d/1vsPmYGTtLBT14FEkj3ah_vICpr3qt87V/view) та
[metadata експорту](https://drive.google.com/file/d/1HlYixXJ9ZqpooNG5mI-T-uXW8OhGpbn5/view).
Generation завершених записів: `1794081756651770183`.

Наступний запуск використовує незмінні наукові конфіги Faster R-CNN у порядку
`photometric_augmentation` → `repeat_factor_sampling` → `input_resolution`.
Кожен експеримент має окремі optimizer/scheduler/checkpoints і п'ять епох.
Спільні параметри: seed 2026, batch 2, stride 5, SGD LR 0.0025, flip 0.5, повна
validation. Higher resolution змінює тільки resize на 800/1280; photometric і sampling
виконуються окремо. YOLO не включено до черги. Це рішення користувача щодо compute;
параметри трьох абляцій не підбирали за Public feedback.

## 14. Public results after the three data ablations

Source: the user's Kaggle submissions screenshot dated 2026-11-08. These are Public
scores, not private results or an independently fetched scorer definition.

| Final epoch-five experiment   | Validation mAP | Kaggle Public |
| ----------------------------- | -------------: | ------------: |
| Faster R-CNN baseline         |       0.174703 |       0.20369 |
| Mild photometric augmentation |       0.178907 |       0.19939 |
| Repeat factor sampling        |       0.182155 |       0.19621 |
| Higher input resolution       |       0.208635 |       0.19425 |
| YOLO11s shared-loop baseline  |       0.154990 |       0.14348 |

All three Faster R-CNN ablations improved canonical validation mAP but reduced Public
score relative to the baseline. This does not establish a causal explanation. Seven
validation videos may be unrepresentative, and the exact competition metric and
ignore semantics are still unconfirmed. We should inspect per-clip and per-class
results rather than treat one aggregate validation improvement as proof of transfer.

The user reported roughly 18 GPU-hours available. Proposed next priorities, not executed:

1. Test slicing-aided inference on saved Faster R-CNN baseline and high-resolution
   weights. Retain a full-frame pass, offset tile boxes into original pixels and merge
   class-aware duplicates. Validate on frames spanning all seven validation clips,
   then evaluate any selected candidate on the full official validation split.
2. If further training is chosen, run a substantially longer YOLO11s fine-tuning recipe
   rather than another five-epoch shared-loop run. Keep eight-class mapping and explicit
   ignored-region treatment. Native training, EMA and schedule choices need a compatible
   integration; do not silently discard ignored regions by switching frameworks.
3. Reserve GPU time for full validation and test export on Kaggle, with verified Drive
   checkpoints and CSVs. Budget allocations are planning limits, not measured runtime.

Slicing is motivated by the small-object problem and the
[SAHI paper](https://arxiv.org/abs/2202.06934). Its published gains on other detectors and
benchmarks do not predict our competition score. A Public target above 0.3 remains a
performance goal, not a guaranteed outcome. No new GPU training was started here.

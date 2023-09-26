from fastapi import FastAPI, UploadFile, File
import os
import cv2
import skimage.io as skio
import skimage.exposure as ske
import skimage.filters as skf
import numpy as np
import skimage.transform as skt
import io
import ultralytics
ultralytics.checks()
from ultralytics import SAM
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO

app = FastAPI()

# Папка для сохранения обработанных и предсказанных изображений
output_dir = "/app/xray"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Загрузка модели YOLO
current_directory = os.path.dirname(os.path.abspath(__file__))
# Путь к модели best_YoloV8n_xray.pt
model_path_yolov8n = os.path.join(current_directory, 'best_YoloV8n_xray.pt')
# Загрузка модели YOLO
model = YOLO(model_path_yolov8n)

# Загрузка модели SAM
model_sam_segment = 'mobile_sam.pt'
model_sam = SAM(model_sam_segment)

@app.post("/predict/")
# Функция для обработки изображения
async def process_image(file: UploadFile = File(...)):
    # Чтение байтов изображения из объекта file
    image_bytes = await file.read()

    # Преобразование байтового потока данных в изображение
    original_image = skio.imread(io.BytesIO(image_bytes))

    # Изменение размера изображения до 950x384 пикселей
    image_resized = skt.resize(original_image, (950, 384), anti_aliasing=True)

    # Дефоггинг
    dehazed_image = ske.adjust_gamma(image_resized, gamma=0.04)

    # Изменение яркости
    brightness_corrected = ske.adjust_gamma(dehazed_image, gamma=0.12)

    # Изменение контраста
    contrast_corrected = ske.adjust_gamma(brightness_corrected, gamma=5.8)

    # Изменение заполняющего света
    fill_light_corrected = ske.adjust_gamma(contrast_corrected, gamma=3.2)

    # Применение фильтра увеличения резкости
    sharpened_image = skf.unsharp_mask(fill_light_corrected, radius=5, amount=5)

    # Генерация имени файла для сохранения обработанного изображения
    filename = file.filename
    processed_filename = f"processed_{filename}"
    save_path = os.path.join(output_dir, processed_filename)

    # Сохранение откорректированного изображения в формате TIF с помощью OpenCV
    cv2.imwrite(save_path, (sharpened_image * 255).astype(np.uint8))

    # Вызов функций predict_objects и segmentation_predict
    predict_path = predict_objects(save_path, output_dir)
    segmentation_path = segmentation_predict(save_path, output_dir)

    return {
        "processed_image": save_path,
        "predicted_image": predict_path,
        "segmentation_image": segmentation_path
    }

# Функция для предсказания объектов на изображении
def predict_objects(image_path, output_dir):
    # Выполняем предсказания на выбранном изображении
    results = model([image_path])

    # Результат предсказания модели YOLO
    for idx, result in enumerate(results):
        boxes = result.boxes  # Boxes object for bbox outputs

        # Определение класса
        classes = {0: 'Опасные', 1: 'Внимание'}  # Пример классов, замените на свои

        # Создайте список для хранения всех осей
        all_axes = []

        # Создание полотна для отображения
        fig = plt.figure(figsize=(10, 5))

        for idx, result in enumerate(results):
            boxes = result.boxes.data.tolist()
            class_labels = result.boxes.data[:, 5].tolist()

            # Добавьте новую ось на полотно
            ax = fig.add_subplot(1, len(results), idx + 1)
            all_axes.append(ax)

            ax.imshow(result.orig_img)  # Использование атрибута orig_img для вывода исходного изображения
            ax.axis('off')  # Без оси

            for box, class_label in zip(boxes, class_labels):
                x1, y1, x2, y2 = box[:4]  # Координаты границ bbox
                class_name = classes[int(class_label)]  # Имя класса по индексу

                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y1, class_name, color='red', fontsize=5, backgroundcolor="white")  # Отобразите класс

        ax.set_title(f"Image {idx}")

        # Генерация имени файла для сохранения результата предсказания
        filename = os.path.basename(image_path)
        prediction_filename = f"prediction_{filename}"

        save_path = os.path.join(output_dir, prediction_filename)

        # Сохранение изображения с результатами предсказания
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, format='tif')
        plt.close(fig)  # Закрываем фигуру, чтобы не отображать ее на экране

    return save_path

# Функция для сегментации объектов на изображении
def segmentation_predict(image_path, output_dir):
    # Загрузка модели YOLO
    results = model.predict(source=image_path, conf=0.25)

    # Загрузка модели SAM
    model_sam = SAM(model_sam_segment)

    # Список для хранения масок сегментации
    segmentation_masks = []

    # Результат предсказания модели YOLO
    for result in results:
        boxes = result.boxes
        bbox_coordinates = boxes.xyxy.tolist()

        for bbox in bbox_coordinates:
            x_min, y_min, x_max, y_max = bbox
            bboxes = [x_min, y_min, x_max, y_max]

            # Предсказание сегментации с помощью модели SAM
            results_list = model_sam.predict(image_path, bboxes=[bboxes])
            results = results_list[0]  # Извлечь результат для первого изображения

            # Извлечение масок сегментации
            masks = results.masks
            segmentation_masks.append(masks)

    # Загрузка изображения
    image = cv2.imread(image_path)

    # Создание копии изображения для наложения сегментации
    overlay = image.copy()
    alpha = 0.5  # Прозрачность сегментации

    # Проход по маскам сегментации и наложение их на изображение
    for masks in segmentation_masks:
        masks_np = masks.data.cpu().numpy()  # Преобразование в массив numpy
        for mask in masks_np:
            mask_np = mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                cv2.drawContours(overlay, [contour], -1, (0, 255, 0), -1)  # Зеленый цвет для контуров

    # Генерация имени файла для сохранения результата сегментации
    filename = os.path.basename(image_path)
    segmentation_filename = f"segmentation_{filename}"
    save_path = os.path.join(output_dir, segmentation_filename)

    # Сохранение изображения с результатами сегментации в формате TIF с помощью OpenCV
    cv2.imwrite(save_path, overlay)

    return save_path

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)












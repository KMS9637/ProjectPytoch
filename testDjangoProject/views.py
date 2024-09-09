from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import ImageSerializer
import torch
from torchvision import transforms, models
from PIL import Image

model_weight_save_path = "testDjangoProject/resnet50_epoch_48_team1_loss_2153_acc_69_52.pth"
num_classes= 5

model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, num_classes)

checkpoint = torch.load(model_weight_save_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint, strict=False)
model.eval()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

class ImageClassificationView(APIView):

    def post(self, request, *args, **kwargs):
        serializer = ImageSerializer(data=request.data)
        if serializer.is_valid():
            image = serializer.validated_data['image']

            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            image = Image.open(image).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(image)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                _, predicted = torch.max(outputs, 1)
                predicted_class_index = predicted.item()
                confidence = probabilities[predicted_class_index].item()

                class_labels = {0: '고양이', 1: '공룡', 2: '강아지',3: '꼬북이',4: '티벳여우'}

                max_confidence, predicted = torch.max(probabilities, 0)
                if max_confidence < 0.5:
                    predicted_class_label = "기타"
                else:
                    predicted_class_label = class_labels.get(predicted.item(), "기타")

                class_confidences = {class_labels[i]: round(probabilities[i].item(), 4) for i in range(num_classes)}

            response_data = {
                'predictedClassLabel': predicted_class_label,
                'confidence': confidence
            }

            return Response(response_data, status=status.HTTP_200_OK)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

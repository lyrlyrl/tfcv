from tfcv.models.resnet import ResNet
from absl import app

def main(_):
    model = ResNet(
        50,
        [224, 224, 3],
        freeze_at=0,
        freeze_bn=True,
        include_top=True,
        pretrained=False
    )
    for v in model.variables:
        print(v.name, v.shape, v.trainable)

if __name__ == '__main__':
    app.run(main)
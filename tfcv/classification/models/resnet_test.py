from tfcv.classification.models.resnet import ResNet
from absl import app

def main(_):
    model = ResNet(
        50,
        [224, 224, 3],
        freeze_at=5,
        freeze_bn=True,
        include_top=False
    )
    model.summary()

if __name__ == '__main__':
    app.run(main)
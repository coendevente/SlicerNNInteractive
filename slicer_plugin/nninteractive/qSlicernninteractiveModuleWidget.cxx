/*==============================================================================

  Program: 3D Slicer

  Portions (c) Copyright Brigham and Women's Hospital (BWH) All Rights Reserved.

  See COPYRIGHT.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

==============================================================================*/

// Qt includes
#include <QDebug>

// Slicer includes
#include "qSlicernninteractiveModuleWidget.h"
#include "ui_qSlicernninteractiveModuleWidget.h"

//-----------------------------------------------------------------------------
class qSlicernninteractiveModuleWidgetPrivate : public Ui_qSlicernninteractiveModuleWidget
{
public:
  qSlicernninteractiveModuleWidgetPrivate();
};

//-----------------------------------------------------------------------------
// qSlicernninteractiveModuleWidgetPrivate methods

//-----------------------------------------------------------------------------
qSlicernninteractiveModuleWidgetPrivate::qSlicernninteractiveModuleWidgetPrivate() {}

//-----------------------------------------------------------------------------
// qSlicernninteractiveModuleWidget methods

//-----------------------------------------------------------------------------
qSlicernninteractiveModuleWidget::qSlicernninteractiveModuleWidget(QWidget* _parent)
  : Superclass(_parent)
  , d_ptr(new qSlicernninteractiveModuleWidgetPrivate)
{
}

//-----------------------------------------------------------------------------
qSlicernninteractiveModuleWidget::~qSlicernninteractiveModuleWidget() {}

//-----------------------------------------------------------------------------
void qSlicernninteractiveModuleWidget::setup()
{
  Q_D(qSlicernninteractiveModuleWidget);
  d->setupUi(this);
  this->Superclass::setup();
}

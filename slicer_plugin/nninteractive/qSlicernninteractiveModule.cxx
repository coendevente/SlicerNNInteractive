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

// nninteractive Logic includes
#include <vtkSlicernninteractiveLogic.h>

// nninteractive includes
#include "qSlicernninteractiveModule.h"
#include "qSlicernninteractiveModuleWidget.h"

//-----------------------------------------------------------------------------
class qSlicernninteractiveModulePrivate
{
public:
  qSlicernninteractiveModulePrivate();
};

//-----------------------------------------------------------------------------
// qSlicernninteractiveModulePrivate methods

//-----------------------------------------------------------------------------
qSlicernninteractiveModulePrivate::qSlicernninteractiveModulePrivate() {}

//-----------------------------------------------------------------------------
// qSlicernninteractiveModule methods

//-----------------------------------------------------------------------------
qSlicernninteractiveModule::qSlicernninteractiveModule(QObject* _parent)
  : Superclass(_parent)
  , d_ptr(new qSlicernninteractiveModulePrivate)
{
}

//-----------------------------------------------------------------------------
qSlicernninteractiveModule::~qSlicernninteractiveModule() {}

//-----------------------------------------------------------------------------
QString qSlicernninteractiveModule::helpText() const
{
  return "This is a loadable module that can be bundled in an extension";
}

//-----------------------------------------------------------------------------
QString qSlicernninteractiveModule::acknowledgementText() const
{
  return "This work was partially funded by NIH grant NXNNXXNNNNNN-NNXN";
}

//-----------------------------------------------------------------------------
QStringList qSlicernninteractiveModule::contributors() const
{
  QStringList moduleContributors;
  moduleContributors << QString("John Doe (AnyWare Corp.)");
  return moduleContributors;
}

//-----------------------------------------------------------------------------
QIcon qSlicernninteractiveModule::icon() const
{
  return QIcon(":/Icons/nninteractive.png");
}

//-----------------------------------------------------------------------------
QStringList qSlicernninteractiveModule::categories() const
{
  return QStringList() << "Examples";
}

//-----------------------------------------------------------------------------
QStringList qSlicernninteractiveModule::dependencies() const
{
  return QStringList();
}

//-----------------------------------------------------------------------------
void qSlicernninteractiveModule::setup()
{
  this->Superclass::setup();
}

//-----------------------------------------------------------------------------
qSlicerAbstractModuleRepresentation* qSlicernninteractiveModule::createWidgetRepresentation()
{
  return new qSlicernninteractiveModuleWidget;
}

//-----------------------------------------------------------------------------
vtkMRMLAbstractLogic* qSlicernninteractiveModule::createLogic()
{
  return vtkSlicernninteractiveLogic::New();
}

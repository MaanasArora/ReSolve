#include <iomanip>
#include <iostream>
#include <sstream>

#include "ExampleHelper.hpp"
#include <resolve/GramSchmidt.hpp>
#include <resolve/LinSolverDirectKLU.hpp>
#include <resolve/LinSolverIterativeFGMRES.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/matrix/io.hpp>
#include <resolve/utilities/params/CliOptions.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/vector/VectorHandler.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>

using namespace ReSolve::constants;
using namespace ReSolve::examples;
using namespace ReSolve;

int testKluRefactor(index_type                     num_systems,
                    const std::vector<index_type>& permutation,
                    const std::string&             matrix_path_name,
                    const std::string&             rhs_path_name,
                    const std::string&             file_extension,
                    double*                        residual_norms,
                    double*                        condition_numbers,
                    int*                           status)
{
  using vector_type = ReSolve::vector::Vector;

  std::string fileId;
  std::string rhsId;
  std::string matrix_file_name_full;
  std::string rhs_file_name_full;

  matrix::Csr*                      A = nullptr;
  LinAlgWorkspaceCpu                workspace;
  ExampleHelper<LinAlgWorkspaceCpu> helper(workspace);
  MatrixHandler                     matrix_handler(&workspace);
  VectorHandler                     vector_handler(&workspace);

  vector_type* vec_rhs = nullptr;
  vector_type* vec_x   = nullptr;

  LinSolverDirectKLU*      KLU = new LinSolverDirectKLU;
  GramSchmidt              GS(&vector_handler, GramSchmidt::CGS2);
  LinSolverIterativeFGMRES FGMRES(&matrix_handler, &vector_handler, &GS);

  for (int i = 0; i < num_systems; ++i)
  {
    int file_index = permutation[i];

    std::ostringstream matname;
    std::ostringstream rhsname;
    matname << matrix_path_name << std::setfill('0') << std::setw(2) << file_index << "." << file_extension;
    rhsname << rhs_path_name << std::setfill('0') << std::setw(2) << file_index << "." << file_extension;
    matrix_file_name_full = matname.str();
    rhs_file_name_full    = rhsname.str();
    std::ifstream mat_file(matrix_file_name_full);
    if (!mat_file.is_open())
    {
      std::cout << "Failed to open file " << matrix_file_name_full << "\n";
      return 1;
    }
    std::ifstream rhs_file(rhs_file_name_full);
    if (!rhs_file.is_open())
    {
      std::cout << "Failed to open file " << rhs_file_name_full << "\n";
      return 1;
    }
    bool is_expand_symmetric = true;
    if (i == 0)
    {
      A = ReSolve::io::createCsrFromFile(mat_file, is_expand_symmetric);

      vec_rhs = ReSolve::io::createVectorFromFile(rhs_file);
      vec_x   = new vector_type(A->getNumRows());
    }
    else
    {
      ReSolve::io::updateMatrixFromFile(mat_file, A);
      ReSolve::io::updateVectorFromFile(rhs_file, vec_rhs);
    }
    mat_file.close();
    rhs_file.close();

    // Now call direct solver
    if (i == 0)
    {
      vec_rhs->setDataUpdated(ReSolve::memory::HOST);
      KLU->setup(A);
      KLU->analyze();
      status[i] = KLU->factorize();
    }
    else
    {
      status[i] = KLU->refactorize();
    }
    condition_numbers[i] = KLU->getMatrixConditionNumber();

    KLU->solve(vec_rhs, vec_x);

    helper.resetSystem(A, vec_rhs, vec_x);
    residual_norms[i]    = helper.getNormRelativeResidual();

    // FGMRES.setup(A);
    // FGMRES.setupPreconditioner("LU", KLU);

    // // If refactorization produced finite solution do iterative refinement
    // if (std::isfinite(helper.getNormRelativeResidual()))
    // {
    //   FGMRES.solve(vec_rhs, vec_x);
    // }
  }

  // now DELETE
  delete A;
  delete KLU;
  delete vec_rhs;
  delete vec_x;
  return 0;
}

int main(int argc, char* argv[])
{
  CliOptions options(argc, argv);
  index_type num_systems = 5;

  std::string matrix_path_name("");
  auto        opt  = options.getParamFromKey("-m");
  matrix_path_name = opt->second;

  std::string rhs_path_name("");
  opt           = options.getParamFromKey("-r");
  rhs_path_name = opt->second;

  std::string file_extension("mtx");

  // permutations of num_systems indices
  std::vector<std::vector<index_type>> permutations = {
      {0, 1, 2, 3, 4},
      {0, 3, 4, 1, 2},
      {4, 3, 2, 1, 0},
      {2, 0, 4, 1, 3},
      {1, 3, 0, 4, 2},
      {3, 4, 1, 0, 2},
      {2, 3, 4, 1, 0},
  };
  double* residual_norms    = new double[num_systems];
  double* condition_numbers = new double[num_systems];
  int*    status            = new int[num_systems];

  for (const auto& perm : permutations)
  {
    std::fill(residual_norms, residual_norms + num_systems, nan(""));
    testKluRefactor(num_systems, perm, matrix_path_name, rhs_path_name, file_extension, residual_norms, condition_numbers, status);

    // Print header
    std::cout << std::setw(10) << "Order" << " | ";
    for (auto i : perm)
    {
      std::cout << std::setw(16) << ("System " + std::to_string(i));
    }
    std::cout << std::endl;

    std::cout << std::string(10, '-') << "-+-";
    for (int i = 0; i < num_systems; ++i)
    {
      std::cout << std::string(16, '-');
    }
    std::cout << std::endl;

    std::cout << std::setw(10) << "Residual" << " | ";
    for (int i = 0; i < num_systems; ++i)
    {
      std::cout << std::setw(16) << std::scientific << std::setprecision(6) << residual_norms[i];
    }
    std::cout << std::endl;

    std::cout << std::setw(10) << "RCond Num." << " | ";
    for (int i = 0; i < num_systems; ++i)
    {
      std::cout << std::setw(16) << std::scientific << std::setprecision(6) << condition_numbers[i];
    }

    std::cout << std::endl
              << std::endl;
  }

  delete[] residual_norms;
  delete[] condition_numbers;
  delete[] status;
  return 0;
}
